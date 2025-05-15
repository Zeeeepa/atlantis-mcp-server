#!/usr/bin/env python3
import logging
import json
import inspect
import asyncio
import os
import importlib
import re
import signal
import sys
import time
import random
import socketio
import argparse
import uuid
import secrets
from utils import clean_filename, format_json_log
from PIDManager import PIDManager
from typing import Any, Callable, Dict, List, Optional, Union
import datetime

# Version
SERVER_VERSION = "0.1.0"

from mcp.server import Server
# Removed websocket_server import - implementing our own handler
from mcp.client.websocket import websocket_client
from mcp.types import (
    Tool,
    TextContent,
    CallToolResult,
    NotificationParams,
    Annotations # Ensure Annotation, ToolErrorAnnotation are NOT imported
# <<< REMOVED Content IMPORT >>>
)
from starlette.applications import Starlette
from starlette.routing import WebSocketRoute, Route
from starlette.websockets import WebSocket, WebSocketDisconnect
from starlette.requests import Request
from starlette.responses import JSONResponse
from mcp.shared.exceptions import McpError # <--- ADD THIS IMPORT

# --- Add dynamic_functions parent to sys.path for imports ---
import sys
import os

try:

    # Get the current timestamp of the dynamic_functions directory
    current_mtime = 0.0
    server_mtime = 0.0

    # Track runtime errors from dynamic functions
    _runtime_errors = {}

    # Get the parent directory of dynamic_functions
    parent_dir = os.path.dirname(FUNCTIONS_DIR)

    # Add the parent directory to sys.path if not already present
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        # Use state.logger if available, otherwise print
        try:
            from state import logger
            logger.info(f"✅ Added '{parent_dir}' to sys.path for dynamic function imports.")
        except ImportError:
            print(f"INFO: Added '{parent_dir}' to sys.path for dynamic function imports.")
except Exception as e:
    print(f"ERROR: Failed to add dynamic_functions parent directory to sys.path: {e}")
# -------------------------------------------------------------

# Import Uvicorn for running the server
import uvicorn

# Import shared state and utilities
from state import (
    logger, # Use the configured logger from state
    HOST, PORT,
    FUNCTIONS_DIR, SERVERS_DIR, is_shutting_down,
    SERVER_REQUEST_TIMEOUT
)

# Cloud server configuration
CLOUD_SERVER_HOST = "ws.projectatlantis.ai"
CLOUD_SERVER_PORT = 3010
CLOUD_SERVER_URL = f"{CLOUD_SERVER_HOST}:{CLOUD_SERVER_PORT}"
CLOUD_SERVICE_NAMESPACE = "/service"  # Socket.IO namespace for service-to-service communication
CLOUD_CONNECTION_RETRY_SECONDS = 5  # Initial delay in seconds
CLOUD_CONNECTION_MAX_RETRIES = 10  # Maximum number of retries before giving up (None for infinite)
CLOUD_CONNECTION_MAX_BACKOFF_SECONDS = 15  # Maximum delay for exponential backoff



# Import color constants directly from ColoredFormatter
from ColoredFormatter import BOLD, RESET, CYAN, BRIGHT_WHITE, PINK

# Import dynamic management classes
from DynamicFunctionManager import DynamicFunctionManager
from DynamicServerManager import DynamicServerManager

# Import our utility module for dynamic functions
import utils

# NOTE: This server uses two different socket protocols:
# 1. Standard WebSockets: When acting as a SERVER to accept connections from node-mcp-client
# 2. Socket.IO: When acting as a CLIENT to connect to the cloud Node.js server
#
# Each server dictates its own protocol, and clients must adapt accordingly.
# - The node-mcp-client connects via standard WebSockets to our server.py
# - Our server.py connects via Socket.IO to the cloud server
# - Both ultimately route to the same MCP handlers in the DynamicAdditionServer class

# Define signal handler for graceful shutdown
def handle_sigint(signum, frame):
    global is_shutting_down
    if not is_shutting_down:
        is_shutting_down = True
        print("\n🐱 Meow! Graceful shutdown in progress... Press Ctrl+C again to force exit! 🐱")
        print("🧹 Cleaning up resources and closing connections...")
        # Signal the cloud connection to close
        if 'cloud_connection' in globals() and cloud_connection is not None:
            logger.info("☁️ Closing cloud server connection...")
            asyncio.create_task(cloud_connection.disconnect())
            pid_manager.remove_pid_file()
    else:
        print("\n🚨 Forced exit! 🚨")
        # Ensure PID file is removed even on force exit
        pid_manager.remove_pid_file()
        sys.exit(1)

# Register signal handler
signal.signal(signal.SIGINT, handle_sigint)
signal.signal(signal.SIGTERM, handle_sigint)

# Initialize PID Manager
pid_manager = PIDManager()

# Check if server is already running
existing_pid = pid_manager.check_server_running()
if existing_pid:
    logger.error(f"❌ MCP SERVER ALREADY RUNNING WITH PID {existing_pid}")
    logger.error(f"❌ If this is incorrect, delete the PID file and try again")
    sys.exit(1)

# Create the PID file
if not pid_manager.create_pid_file():
    logger.error("❌ Failed to create PID file! Exiting...")
    sys.exit(1)

# --- File Watcher Setup ---

import threading # Added for watcher thread
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class DynamicConfigEventHandler(FileSystemEventHandler):
    """Handles file system events in dynamic_functions and dynamic_servers directories."""
    def __init__(self, mcp_server, loop):
        self.mcp_server = mcp_server
        self.loop = loop
        self._debounce_timer = None
        self._debounce_interval = 1.0 # seconds
        self._last_triggered_path = None # Store path for _do_reload

    def _trigger_reload(self, event_path):
        # Check if the change is relevant (Python file in functions dir or JSON file in servers dir)
        is_function_change = event_path.endswith(".py") and os.path.dirname(event_path) == FUNCTIONS_DIR
        is_server_change = event_path.endswith(".json") and os.path.dirname(event_path) == SERVERS_DIR

        if not is_function_change and not is_server_change:
            # logger.debug(f"Ignoring irrelevant change: {event_path}")
            return

        change_type = "function" if is_function_change else "server configuration"
        logger.info(f"🐍 Change detected in dynamic {change_type}: {os.path.basename(event_path)}. Debouncing...")

        # Store the path that triggered this potential reload
        self._last_triggered_path = event_path

        # Debounce: Cancel existing timer if a new event comes quickly
        if self._debounce_timer:
            self._debounce_timer.cancel()

        # Schedule the actual reload after a short delay
        # Pass the event path to the target function
        self._debounce_timer = threading.Timer(self._debounce_interval, self._do_reload, args=[event_path])
        self._debounce_timer.start()

    def _do_reload(self, event_path: Optional[str] = None):
        # Use the path passed from the timer, or the last stored one as fallback
        path_to_process = event_path or self._last_triggered_path
        if not path_to_process:
            logger.warning("⚠️ _do_reload called without a valid event path.")
            return

        logger.info(f"⏰ Debounce finished for {os.path.basename(path_to_process)}. Processing file change.")

        # Clear the cache - Must run in the main event loop
        async def _process_file_change(file_path: str):
            # --- Invalidate ALL Dynamic Function Runtime Caches ---
            # Check if the change was in the functions or servers directory
            is_function_change = file_path.endswith(".py") and os.path.dirname(file_path) == FUNCTIONS_DIR
            is_server_change = file_path.endswith(".json") and os.path.dirname(file_path) == SERVERS_DIR

            if is_function_change:
                logger.info(f"⚡ File watcher triggering flush of all dynamic function runtime caches due to change in {os.path.basename(file_path)}.")
                try:
                    await self.mcp_server.function_manager.invalidate_all_dynamic_module_cache()
                except Exception as e:
                    logger.error(f"❌ Error invalidating all dynamic modules cache from file watcher: {e}")
            # --- End Runtime Cache Invalidation ---

            # --- Existing Tool List Cache Clearing & Notification ---
            # This still runs regardless of whether it was a function or server config change
            logger.info(f"🧹 Clearing tool list cache on server due to file change in {os.path.basename(file_path)}.")
            self.mcp_server._cached_tools = None
            self.mcp_server._last_functions_dir_mtime = None # Reset functions mtime to force reload
            self.mcp_server._last_servers_dir_mtime = None   # Reset servers mtime to force reload
            # Notify clients
            if hasattr(self.mcp_server, '_notify_tool_list_changed'):
                try:
                    # Pass placeholder args as the watcher doesn't know specifics
                    await self.mcp_server._notify_tool_list_changed(change_type="unknown", tool_name="unknown/file_watcher")
                except Exception as e:
                    # Log the specific error from the notification call
                    logger.error(f"❌ Failed to notify clients after file change: {e}")
            else:
                logger.warning("⚠️ Server object lacks _notify_tool_list_changed method.")
            # --- End Tool List Cache Clearing ---

        # Schedule the coroutine to run in the event loop from this thread
        if self.loop.is_running():
             asyncio.run_coroutine_threadsafe(_process_file_change(path_to_process), self.loop)
        else:
             logger.warning("⚠️ Event loop not running, cannot process file change for {os.path.basename(path_to_process)}.")

    def on_modified(self, event):
        if not event.is_directory:
            self._trigger_reload(event.src_path)

    def on_created(self, event):
        if not event.is_directory:
            self._trigger_reload(event.src_path)

    def on_deleted(self, event):
        if not event.is_directory:
            self._trigger_reload(event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            # Trigger reload for both source and destination paths
            self._trigger_reload(event.src_path)
            self._trigger_reload(event.dest_path)

# --- End File Watcher Setup ---

# Create an MCP server with proper MCP protocol handling
class DynamicAdditionServer(Server):
    """MCP server that provides an addition tool and supports dynamic function registration"""

    def __init__(self):
        super().__init__("Dynamic Function Server")
        self.websocket_connections = set()  # Store active WebSocket connections
        self.service_connections = {} # Store active Service connections (e.g. cloud)
        # TODO: Add prompts and resources
        self._cached_tools: Optional[List[Tool]] = None # Cache for tool list
        self._last_functions_dir_mtime: float = 0.0 # Timestamp for cache invalidation
        self._last_servers_dir_mtime: float = 0.0 # Timestamp for dynamic servers cache invalidation
        self._last_active_server_keys: Optional[set] = None # Store active server keys for cache invalidation
        self._server_configs: Dict[str, dict] = {} # Store server configurations

        # Initialize the dynamic function and server managers
        self.function_manager = DynamicFunctionManager(FUNCTIONS_DIR)
        self.server_manager = DynamicServerManager(SERVERS_DIR)

        # Register tool handlers using SDK decorators
        # These now wrap the actual logic methods defined below

        @self.list_tools()
        async def handle_list_tools() -> list[Tool]:
            """SDK Handler for tools/list"""
            logger.info(f"🚀 Processing MCP tool list request")
            # Pass context indicating this call is from the SDK handler (likely a direct request)
            return await self._get_tools_list(caller_context="handle_list_tools_sdk")

        @self.list_prompts()
        async def handle_list_prompts() -> list:
            """SDK Handler for prompts/list"""
            return await self._get_prompts_list()

        @self.list_resources()
        async def handle_list_resources() -> list:
            """SDK Handler for resources/list"""
            return await self._get_resources_list()

        @self.call_tool()
        async def handle_call_tool(name: str, args: dict) -> list[TextContent]:
            """SDK Handler for tools/call"""
            return await self._execute_tool(name=name, args=args)

    # Initialization for function discovery
    async def initialize(self, params={}):
        """Initialize the server, sending a toolsList notification with initial tools"""
        logger.info(f"{CYAN}🔧 === ENTERING SERVER INITIALIZE METHOD ==={RESET}")
        logger.info(f"🚀 Server initialized with version {SERVER_VERSION}")

        # Set the server instance in utils module for client logging
        utils.set_server_instance(self)
        logger.info("🔌 Dynamic functions utility module initialized")

        tools_list = await self._get_tools_list(caller_context="initialize_method")
        logger.info(f"{CYAN}🔧 Server initialize method completed.{RESET}")
        # Return required InitializeResult fields
        return {
            "protocolVersion": params.get("protocolVersion"),
            "capabilities": params.get("capabilities"),
            "serverInfo": {"name": self.name, "version": SERVER_VERSION}
        }

    async def _get_tools_list(self, caller_context: str = "unknown") -> list[Tool]:
        """Core logic to return a list of available tools"""
        logger.info(f"{BRIGHT_WHITE}📋 === GETTING TOOL LIST (Called by: {caller_context}) ==={RESET}")

        # --- Caching Logic ---
        try:
            # Ensure the directory exists before checking mtime
            if not os.path.exists(FUNCTIONS_DIR):
                 os.makedirs(FUNCTIONS_DIR, exist_ok=True)
                 current_mtime = os.path.getmtime(FUNCTIONS_DIR)
            else:
                 current_mtime = os.path.getmtime(FUNCTIONS_DIR)

            # Ensure the dynamic servers directory exists and get its mtime
            if not os.path.exists(SERVERS_DIR):
                os.makedirs(SERVERS_DIR, exist_ok=True)
                server_mtime = os.path.getmtime(SERVERS_DIR)
            else:
                server_mtime = os.path.getmtime(SERVERS_DIR)

            dirs_changed = current_mtime != self._last_functions_dir_mtime or server_mtime != self._last_servers_dir_mtime
            # First identify running vs. non-running servers            # Get running servers using proper accessor
            running_servers = await self.server_manager.get_running_servers()
            current_active_keys = set(running_servers)
            servers_changed = current_active_keys != self._last_active_server_keys

            # Add 'and False' to always invalidate cache for now
            # Use current_active_keys here too
            if not dirs_changed and not servers_changed and self._cached_tools is not None and False:
                logger.info(f"⚡️ USING CACHED TOOL LIST (Dirs unchanged - func mtime: {current_mtime}, server mtime: {server_mtime}; Active Servers unchanged: {self._last_active_server_keys})")
                return list(self._cached_tools)
            # Log reason for regeneration
            reason = []
            if dirs_changed:
                reason.append(f"Dirs changed (func mtime: {current_mtime} vs {self._last_functions_dir_mtime}, server mtime: {server_mtime} vs {self._last_servers_dir_mtime})")
            if servers_changed:
                reason.append(f"Active servers changed ({running_servers} vs {self._last_active_server_keys})")
            if self._cached_tools is None:
                 reason.append("Cache empty")
            logger.info(f"🔄 Cache invalid ({', '.join(reason)}). REGENERATING TOOL LIST")

        except FileNotFoundError as e:
             logger.error(f"❌ Error checking FUNCTIONS_DIR mtime: {e}. Proceeding without cache.")
             current_mtime = time.time() # Use current time to force regeneration

        # Start with our built-in tools
        tools_list = [
            Tool(
                name="_function_set",
                description="Sets the content of a dynamic Python function", # Updated description
                inputSchema={
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "The Python source code containing a single function definition."}
                    },
                    "required": ["code"] # Only code is required now
                }
            ),
            Tool( # Add definition for get_tool_code
                name="_function_get",
                description="Gets the Python source code for a dynamic function",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "The name of the function to get code for"}
                    },
                    "required": ["name"]
                }
            ),
            Tool( # Add definition for remove_dynamic_tool
                name="_function_remove",
                description="Removes a dynamic Python function",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "The name of the function to remove"}
                    },
                    "required": ["name"]
                }
            ),
            Tool( # Add definition for add_placeholder_function
                name="_function_add",
                description="Adds a new, empty placeholder Python function with the given name.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "The name to register the new placeholder function under."}
                    },
                    "required": ["name"]
                }
            ),
            Tool(
                name="_server_get",
                description="Gets the configuration JSON for a server",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    },
                    "required": ["name"]
                }
            ),
            Tool(
                name="_server_add",
                description="Adds a new MCP server configuration",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    },
                    "required": ["name"]
                }
            ),
            Tool(
                name="_server_remove",
                description="Removes an MCP server configuration",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    },
                    "required": ["name"]
                }
            ),
            Tool(
                name="_server_set",
                description="Sets (adds or updates) an MCP server configuration",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "config": {
                            "type": "object",
                            "description": "Server config JSON, must contain 'mcpServers' key with the server name as its child.",
                            "properties": {
                                "mcpServers": {
                                    "type": "object",
                                    "description": "A dictionary where the key is the server name.",
                                    # We don't strictly enforce the structure *within* the named server config here,
                                    # relying on server_manager's validation, but describe the expectation.
                                    "additionalProperties": {
                                         "type": "object",
                                         "properties": {
                                            "command": {"type": "string"},
                                            "args": {"type": "array", "items": {"type": "string"}},
                                            "env": {"type": "object"},
                                            "cwd": {"type": "string"}
                                         },
                                         "required": ["command"]
                                    }
                                }
                            },
                            "required": ["mcpServers"]
                        }
                    },
                    "required": ["config"] # Only config is required top-level
                }
            ),
            Tool(
                name="_server_validate",
                description="Validates an MCP server configuration",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    },
                    "required": ["name"]
                }
            ),
            Tool(
                name="_server_start",
                description="Starts a managed MCP server background task using its configuration name.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "The name of the server config to start."}
                    },
                    "required": ["name"]
                }
            ),
            Tool(
                name="_server_stop",
                description="Stops a managed MCP server background task by its configuration name.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "The name of the server config to stop."}
                    },
                    "required": ["name"]
                }
            ),
            Tool(
                name="_server_get_tools",
                description="Gets the list of tools from a specific *running* managed server.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    },
                    "required": ["name"]
                }
            ),

        ]
        # Log the static tools being included
        static_tool_names = [tool.name for tool in tools_list]
        logger.info(f"🔩 INCLUDING {len(static_tool_names)} STATIC TOOLS: {', '.join(static_tool_names)}")

        # Scan the FUNCTIONS_DIR directory for dynamic functions
        try:
            # Ensure the functions directory exists
            os.makedirs(FUNCTIONS_DIR, exist_ok=True)

            # Get all Python files in the directory
            function_files = [f for f in os.listdir(FUNCTIONS_DIR) if f.endswith('.py')]
            logger.info(f"📝 FOUND {len(function_files)} POTENTIAL DYNAMIC FUNCTIONS")

            # For each Python file, create a Tool entry
            for file_name in function_files:
                tool_name_from_file = os.path.splitext(file_name)[0]
                try:
                    # Skip if this seems to be a utility file
                    if tool_name_from_file.startswith('_') or tool_name_from_file == '__init__' or tool_name_from_file == '__pycache__':
                        continue

                    # Validate function syntax without actually loading it
                    validation_result = await self.function_manager.function_validate(tool_name_from_file)
                    is_valid = validation_result.get('valid', False)
                    error_message = validation_result.get('error')
                    function_info = validation_result.get('function_info') # Should be a dict if valid

                    # --- Create Tool --- #
                    tool_name = tool_name_from_file
                    tool_description = f"Dynamic function: {tool_name_from_file}"
                    tool_input_schema = {"type": "object", "properties": {}}
                    tool_annotations = {}

                    tool_annotations["type"] = "function"

                    if is_valid and function_info:
                         # Use extracted info if valid and available
                         tool_name = function_info.get('name', tool_name_from_file) # Use AST name, fallback to filename
                         tool_description = function_info.get('description', f"Dynamic function '{tool_name}'")
                         tool_input_schema = function_info.get('inputSchema', {"type": "object", "properties": {}})
                         tool_annotations["validationStatus"] = "VALID"
                    elif is_valid and not function_info:
                         # Valid syntax but failed to extract info (should ideally not happen)
                         tool_description = f"Dynamic function: {tool_name_from_file} (Details unavailable)"
                         tool_input_schema = {"type": "object", "description": "Could not parse arguments."}
                         tool_annotations["validationStatus"] = "VALID_SYNTAX_UNKNOWN_STRUCTURE"
                    else:
                         # Invalid syntax
                         tool_description = f"Dynamic function: {tool_name_from_file} (INVALID)"
                         tool_input_schema = {"type": "object", "description": "Function has syntax errors."}
                         tool_annotations["validationStatus"] = "INVALID"
                         if error_message:
                             tool_annotations["errorMessage"] = error_message

                    # Add runtime error message if present in cache
                    if tool_name_from_file in _runtime_errors:
                        tool_annotations["runtimeError"] = _runtime_errors[tool_name_from_file]

                    # Add server config load error if present in cache
                    if tool_name_from_file in self.server_manager._server_load_errors:
                        tool_annotations["loadError"] = self.server_manager._server_load_errors[tool_name_from_file]

                    # Add common annotations
                    try:
                        tool_annotations["lastModified"] = datetime.datetime.fromtimestamp(
                            os.path.getmtime(os.path.join(FUNCTIONS_DIR, file_name))
                        ).isoformat()
                    except Exception:
                        pass # Ignore if file stat fails

                    # Create and add the tool object
                    tool_obj = Tool(
                        name=tool_name,
                        description=tool_description,
                        inputSchema=tool_input_schema,
                        annotations=tool_annotations
                    )
                    tools_list.append(tool_obj)
                    logger.debug(f"📝 Added dynamic tool: {tool_name}, valid: {is_valid}")

                except Exception as e:
                    logger.warning(f"⚠️ Error processing potential tool file {file_name}: {str(e)}")
                    # Add a placeholder indicating the error
                    tools_list.append(Tool(
                        name=tool_name_from_file,
                        description=f"Error loading dynamic function: {str(e)}",
                        inputSchema={"type": "object"},
                        annotations={"validationStatus": "ERROR_LOADING"}
                    ))
                    continue

        except Exception as e:
            logger.error(f"❌ Error scanning for dynamic functions: {str(e)}")
            # Continue with just the built-in tools

        # --- DEBUG: Log current ACTIVE_SERVER_TASKS state before processing servers ---
        # Access ACTIVE_SERVER_TASKS via module namespace
        logger.debug(f"🔧 _get_tools_list: Current active_server_tasks state: {self.server_manager.active_server_tasks!r}")

        # Scan dynamic servers
        servers_found = []
        self._server_configs = {} # Reset server configs
        server_statuses = {}
        try:
            # server_list now returns List[TextContent]
            # Use server_manager.server_list if it wasn't imported specifically
            server_list_results = await self.server_manager.server_list() # Using server_manager instance
            logger.info(f"📝 FOUND {len(server_list_results)} MCP server configs")
            for server_name in server_list_results:
                # server_list_results is now a list of strings, not TextContent objects
                # so we can use the server_name directly
                servers_found.append(server_name)
                # First identify running vs. non-running servers using proper accessor
                is_running = await self.server_manager.is_server_running(server_name)

                # Check if server is in failed state in active_server_tasks
                task_info = self.server_manager.active_server_tasks.get(server_name, {})
                if task_info and task_info.get('status') == 'failed':
                    status = "failed"  # Mark explicitly as failed
                else:
                    status = "running" if is_running else "stopped" # Normal running/stopped state

                # --- AUTO-START LOGIC --- #
                '''
                if status == "stopped":
                    logger.info(f"⚙️ Server '{server_name}' is stopped. Attempting auto-start during tool list generation...")
                    try:
                        # Call server_start logic internally
                        start_result = await self.server_manager.server_start({'name': server_name}, self)
                        logger.info(f"✅ Auto-start initiated for server '{server_name}'. Result: {start_result}")
                        # Re-check status *after* attempting start
                        is_running = await self.server_manager.is_server_running(server_name)
                        status = "running" if is_running else "stopped"
                        if status == "stopped":
                             logger.warning(f"⚠️ Auto-start attempt for '{server_name}' did not result in an active task. Status remains 'stopped'.")
                        else:
                             logger.info(f"👍 Server '{server_name}' successfully auto-started. Status is now 'running'.")
                    except Exception as start_err:
                        logger.error(f"❌ Failed to auto-start server '{server_name}' during tool list generation: {start_err}", exc_info=False) # Less noisy log
                        status = "stopped"
                # --- END AUTO-START LOGIC --- #
                '''
                server_statuses[server_name] = status # Store final status

                # Always try to get config and add server entry (even if stopped)
                try:
                    # Get server config
                    config = await self.server_manager.server_get(server_name)
                    self._server_configs[server_name] = config # Populate instance variable

                    # Create annotations object
                    annotations = {}
                    annotations["type"] = "server" # Mark this tool entry as representing a server config
                    annotations["serverConfig"] = config or {}
                    annotations["runningStatus"] = status # Add status from outer loop

                    # Try to get file modified time
                    try:
                        server_file = os.path.join(SERVERS_DIR, f"{server_name}.json")
                        mtime = os.path.getmtime(server_file)
                        annotations["lastModified"] = datetime.datetime.fromtimestamp(mtime).isoformat()
                    except Exception as me:
                        logger.warning(f"⚠️ Could not get mtime for server '{server_name}': {me}")

                    # Define task_info FIRST, outside any status checks
                    task_info = self.server_manager.active_server_tasks.get(server_name)

                    # Better check for running status - don't rely on 'status' field which might not be set
                    # Instead check if task exists, is not done, and session exists
                    is_running = (
                        task_info is not None and
                        'task' in task_info and
                        not task_info['task'].done() and
                        'session' in task_info and
                        task_info['session'] is not None
                    )
                    status = "running" if is_running else "stopped"

                    if is_running:
                        logger.debug(f"🔄 Server '{server_name}' is running (task active and session present)")
                    else:
                        logger.debug(f"⏹️ Server '{server_name}' is not running")
                        if task_info:
                            logger.debug(f"📊 Task info: task present={('task' in task_info)}, session present={('session' in task_info and task_info['session'] is not None)}")
                            if 'task' in task_info and task_info['task'].done():
                                logger.debug(f"⚠️ Task for '{server_name}' is marked as done")

                    # Update annotations with final status
                    annotations["runningStatus"] = status

                    # Get timestamp if server is running
                    if status == "running":
                        # Try to get started_at from task_info
                        start_time = task_info.get('started_at')

                        # If not found, and we know it's running, set it now with current time
                        if not start_time or not isinstance(start_time, datetime.datetime):
                            logger.debug(f"🕒 Setting started_at timestamp for '{server_name}' to current time")
                            start_time = datetime.datetime.now(datetime.timezone.utc)
                            # Save it in ACTIVE_SERVER_TASKS for future use
                            if task_info:
                                task_info['started_at'] = start_time

                        # Add timestamp to annotations
                        logger.debug(f"🕒 Using started_at timestamp for '{server_name}': {start_time}")
                        annotations["started_at"] = start_time.isoformat()
                    else:
                        logger.debug(f"📚 No started_at timestamp set for non-running server '{server_name}'")

                        # If this is a stoppped server with a start time in the past, include it anyway
                        # This lets us preserve timestamps for servers that were running but are now stopped
                        past_time = task_info.get('started_at') if task_info else None
                        if past_time and isinstance(past_time, datetime.datetime):
                            logger.debug(f"📚 Found historical started_at timestamp for '{server_name}': {past_time}")
                            annotations["started_at"] = past_time.isoformat()

                        # Always check ALL error sources thoroughly
                    error_msg = None
                    task_info = self.server_manager.active_server_tasks.get(server_name, {})

                    # STEP 1: Log ALL possible error sources
                    logger.info(f"🔍 SERVER STATUS CHECK FOR '{server_name}':")
                    logger.info(f"  - Current status: {status}")
                    logger.info(f"  - Task info available: {task_info is not None}")
                    logger.info(f"  - Has error in task_info: {'error' in task_info if task_info else False}")
                    logger.info(f"  - Error in _server_load_errors: {server_name in self.server_manager._server_load_errors}")

                    # STEP 2: Always show the full error contents from both sources
                    if task_info and 'error' in task_info:
                        task_error = task_info['error']
                        logger.info(f"  - TASK ERROR: {task_error}")

                    if server_name in self.server_manager._server_load_errors:
                        load_error = self.server_manager._server_load_errors[server_name]
                        logger.info(f"  - LOAD ERROR: {load_error}")

                    # STEP 3: Check for failures
                    # First from failed task status
                    if status == 'failed' or (task_info and task_info.get('status') == 'failed'):
                        logger.info(f"🚨 Server '{server_name}' has FAILED status")
                        # Update status to reflect the failure
                        status = "failed"
                        annotations["runningStatus"] = "failed"

                        # Get error from task_info first (most accurate)
                        if task_info and 'error' in task_info:
                            error_msg = task_info['error']
                            logger.info(f"🚨 Using error from task_info: {error_msg}")
                        # Fall back to _server_load_errors
                        elif server_name in self.server_manager._server_load_errors:
                            error_msg = self.server_manager._server_load_errors[server_name]
                            logger.info(f"🚨 Using error from _server_load_errors: {error_msg}")
                        # Last resort generic error
                        else:
                            error_msg = "Server failed to initialize (unknown error)"
                            logger.info(f"⚠️ Using generic error: {error_msg}")

                    # Even for non-failed servers, check for errors
                    elif server_name in self.server_manager._server_load_errors:
                        error_msg = self.server_manager._server_load_errors[server_name]
                        logger.info(f"🚨 Found error for '{server_name}' in load errors: {error_msg}")
                        # If we found an error but status isn't failed, update it
                        status = "failed"
                        annotations["runningStatus"] = "failed"
                    elif task_info and 'error' in task_info:
                        error_msg = task_info['error']
                        logger.info(f"🚨 Found error for '{server_name}' in task info: {error_msg}")
                        # If we found an error but status isn't failed, update it
                        status = "failed"
                        annotations["runningStatus"] = "failed"

                    # Add error info to annotations if found
                    if error_msg:
                        logger.info(f"📣 Final error for '{server_name}': {error_msg}")
                        # Add to both error fields for maximum visibility
                        annotations["loadError"] = error_msg
                        annotations["error"] = error_msg

                    # Simplified logging that only uses defined variables
                    if task_info:
                        logger.debug(f"🔧 _get_tools_list: Preparing Tool for '{server_name}': task_info exists={task_info is not None}, status='{status}'")
                    else:
                        logger.warning(f"🔧 _get_tools_list: Tool '{server_name}' not running")

                    # Create tool with correct parameters
                    description = f"MCP server: {server_name}"
                    # do not append error to the description

                    server_tool = Tool(
                        name=server_name,
                        description=description,
                        inputSchema={"type": "object"}, # Use inputSchema (camelCase), not input_schema
                        annotations=annotations # Use annotations object that contains started_at if applicable
                    )

                    tools_list.append(server_tool)
                    logger.debug(f"📝 Added dynamic server config entry: {server_name} (Status: {status})")
                except Exception as se:
                    logger.warning(f"⚠️ Error processing MCP server config '{server_name}': {se}")
                    tools_list.append(Tool(
                        name=server_name,
                        description=f"Error loading MCP server config: {se}",
                        inputSchema={"type": "object"},
                        annotations={
                            "validationStatus": "ERROR_LOADING_SERVER",
                            "runningStatus": status # Still show status even if config load failed
                            }
                    ))
        except Exception as ee:
            logger.error(f"❌ Error scanning for dynamic servers: {ee}")

        # --- Fetch Tools from RUNNING Servers Concurrently --- #
        running_servers = await self.server_manager.get_running_servers()
        if running_servers:
            logger.info(f"📡 Fetching tools from {len(running_servers)} running servers: {running_servers}")
            fetch_tasks = []
            task_to_server = {}
            for server_name in running_servers:
                # Use the imported get_server_tools function
                task = asyncio.create_task(self.server_manager.get_server_tools(server_name))
                fetch_tasks.append(task)
                task_to_server[task] = server_name

            # Wait for all tasks to complete
            gather_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

            # Process results
            for task, result in zip(fetch_tasks, gather_results):
                server_name = task_to_server[task]
                if isinstance(result, Exception):
                    logger.error(f"❌ Failed to fetch tools from running server '{server_name}': {result}")
                    # Optionally update the server's entry in tools_list with an error annotation
                    for tool in tools_list:
                        if tool.name == server_name and tool.annotations.get("type") == "server":
                            tool.annotations["toolFetchError"] = str(result)
                            #tool.description += f" (Error fetching tools: {result})"
                            break
                elif isinstance(result, list): # Could be list[dict] or list[Tool]
                    logger.info(f"✅ Fetched {len(result)} tools from server '{server_name}'")
                    for tool_item in result:
                        # Handle both Tool objects and dictionaries
                        if hasattr(tool_item, 'name') and hasattr(tool_item, 'description') and hasattr(tool_item, 'inputSchema'):
                            # It's a Tool object
                            original_name = tool_item.name
                            original_description = tool_item.description
                            original_schema = tool_item.inputSchema
                            original_annotations = getattr(tool_item, 'annotations', {})
                        elif isinstance(tool_item, dict):
                            # It's a dictionary
                            original_name = tool_item.get('name')
                            if not original_name:
                                logger.warning(f"⚠️ Received tool dictionary from {server_name} with missing name: {tool_item}")
                                continue
                            original_description = tool_item.get('description', original_name)
                            original_schema = tool_item.get('inputSchema', {"type": "object"})
                            original_annotations = tool_item.get('annotations', {})
                        else:
                            logger.warning(f"⚠️ Received unsupported item type from {server_name}: {type(tool_item)} - {tool_item}")
                            continue

                        # We've already extracted the data in the previous block

                        new_tool_name = f"{server_name}.{original_name}"
                        #new_description = f"[From {server_name}] {original_description}"
                        new_description = original_description
                        new_annotations = {
                            **original_annotations,
                            "originServer": server_name,
                            "type": "server_tool" # Mark as tool provided by a dynamic server
                        }

                        # Create the new tool entry using extracted data
                        new_tool = Tool(
                            name=new_tool_name,
                            description=new_description,
                            inputSchema=original_schema,
                            annotations=new_annotations
                        )
                        tools_list.append(new_tool)
                        logger.debug(f"  -> Added tool from server: {new_tool_name}")
                else:
                     logger.warning(f"❓ Unexpected result type from get_server_tools for '{server_name}': {type(result)}")

        logger.info(f"📝 FOUND {len(tools_list)} TOTAL TOOLS (including from servers)")

        # --- Update Cache --- #
        self._cached_tools = list(tools_list) # Store a copy
        self._last_functions_dir_mtime = current_mtime
        self._last_servers_dir_mtime = server_mtime
        # First identify running vs. non-running servers using proper accessor
        self._last_active_server_keys = set(await self.server_manager.get_running_servers()) # Store active server keys
        logger.info(f"💾 TOOL LIST (functions ts: {current_mtime}; servers ts: {server_mtime})")

        return tools_list

    async def _get_prompts_list(self) -> list:
        """Core logic to return a list of available prompts (empty stub)"""
        # Currently no prompts supported
        return []

    async def _get_resources_list(self) -> list:
        """Core logic to return a list of available resources (empty stub)"""
        # Currently no resources supported
        return []

    async def send_client_log(self,
                              level: str,
                              data: Any,
                              logger_name: str = None,
                              request_id: str = None,
                              client_id: str = None,
                              seq_num: Optional[int] = None,
                              entry_point_name: Optional[str] = None # <-- Add new param
                              ):
        """Send a log message notification to connected clients using direct WebSocket communication.

        Args:
            level: The log level ("debug", "info", "warning", "error")
            data: The log message content (can be string or structured data)
            logger_name: Optional name to identify the specific function emitting the log
            request_id: Optional ID of the original request for client-side correlation
            client_id: Optional client identifier for routing the message
            seq_num: Optional sequence number for client-side ordering
            entry_point_name: Optional name of the top-level function originally called
        """
        try:
            # Normalize level to uppercase for consistency
            level = level.upper()

            # Create a simple JSON-RPC notification structure
            params = {
                "level": level,
                "data": data,
                "logger": logger_name or "unknown_caller", # The immediate caller
                "requestId": request_id,
                "entryPoint": entry_point_name or "unknown_entry_point" # The original entry point
            }

            # Add seqNum if provided
            if seq_num is not None:
                params["seqNum"] = seq_num

            notification = {
                "jsonrpc": "2.0",
                "method": "notifications/message",
                "params": params # Use the params dict built above
            }

            # Get the global tracking collections
            global active_websockets, client_connections, current_request_client_id

            # If no specific client_id was provided, try to use the one from the current request
            if client_id is None and 'current_request_client_id' in globals():
                client_id = current_request_client_id

            # ONLY send to the specific client that made the request - NO broadcasting
            if client_id and client_id in client_connections:
                # Convert to JSON string
                import json
                notification_json = json.dumps(notification)

                # Enhanced logging for client log routing
                logger.info(f"📋 CLIENT LOG ROUTING: Sending to client_id={client_id}, request_id={request_id}")
                logger.info(f"📋 KNOWN CLIENTS: {list(client_connections.keys())}")
                # Log the notification for debugging (now includes client_id if added)
                logger.debug(f"Sending client log notification: {notification_json}")

                client_info = client_connections[client_id]
                client_type = client_info.get("type")
                connection = client_info.get("connection")

                if client_type == "websocket" and connection:
                    try:
                        await connection.send_text(notification_json)
                        logger.debug(f"📢 Sent notification to specific client: {client_id}")
                    except Exception as e:
                        logger.warning(f"Failed to send to client {client_id}: {e}")

                elif client_type == "cloud" and connection and connection.is_connected:
                    try:
                        await connection.send_message('mcp_notification', notification)
                        logger.debug(f"☁️ Sent notification to cloud client: {client_id}")
                    except Exception as e:
                        logger.warning(f"Failed to send to cloud client {client_id}: {e}")
            else:
                logger.warning(f"Cannot send client log: no valid client_id provided or client not found: {client_id}")
                # Log the client connections we know about for debugging
                logger.debug(f"Known client connections: {list(client_connections.keys())}")

        except Exception as e:
            # Don't let logging errors affect the main operation
            logger.error(f"❌ Error sending direct client log notification: {str(e)}")
            import traceback
            logger.debug(f"Log notification error details: {traceback.format_exc()}")
            # We intentionally don't re-raise here


    async def _execute_tool(self, name: str, args: dict, client_id: str = None, request_id: str = None, user: str = None) -> list[TextContent]:
        """Core logic to handle a tool call. Ensures result is List[TextContent(type='text')]"""
        logger.info(f"🔧 EXECUTING TOOL: {name}")
        logger.debug(f"WITH ARGUMENTS: {args}")
        if user:
            logger.debug(f"CALLED BY USER: {user}")
        # ---> ADDED: Log entry and raw args
        logger.debug(f"---> _execute_tool ENTERED. Name: '{name}', Raw Args: {args!r}") # <-- ADD THIS LINE

        try:
            result_raw = None # Initialize raw result variable
            # Handle built-in tool calls
            if name == "_function_set":
                logger.debug(f"---> Calling built-in: function_set") # <-- ADD THIS LINE
                # function_set now returns (extracted_name, result_messages)
                extracted_name, result_messages = await self.function_manager.function_set(args, self)
                result_raw = result_messages # Use the messages returned by function_set
                if extracted_name:
                    # Notify only if function_set successfully extracted a name
                    await self._notify_tool_list_changed(change_type="updated", tool_name=extracted_name)
            elif name == "_function_get":
                logger.debug(f"---> Calling built-in: get_function_code") # <-- ADD THIS LINE
                function_name = args.get('name')
                code = await self.function_manager._fs_load_code(function_name)
                if code is None:
                    result_raw = [TextContent(type="text", text=f"Error: Function '{function_name}' not found or could not be loaded.")]
                else:
                    result_raw = [TextContent(type="text", text=code)]
            elif name == "_function_remove":
                # Remove function
                func_name = args.get("name")
                if not func_name:
                    raise ValueError("Missing required parameter: name")

                logger.debug(f"---> Calling built-in: function_remove for '{func_name}'") # <-- ADD THIS LINE

                # Check if function exists before attempting to remove
                function_path = os.path.join(FUNCTIONS_DIR, f"{func_name}.py")
                if not os.path.exists(function_path):
                    # Create annotation dict for 'function does not exist' error
                    error_message = f"Function '{func_name}' does not exist or was already removed."
                    error_annotations = {
                        "tool_error": {"tool_name": name, "message": error_message}
                    }
                    result_raw = [TextContent(type="text", text=error_message, annotations=error_annotations)]
                else: # <-- ADD else block
                    # Remove the function using dynamic_function_manager.function_remove (raise error on failure)
                    removed = await self.function_manager.function_remove(func_name)
                    if removed:
                        try:
                            await self._notify_tool_list_changed(change_type="removed", tool_name=func_name) # Pass params
                        except Exception as e:
                            logger.error(f"Error sending tool notification after removing {func_name}: {str(e)}")
                        result_raw = [TextContent(type="text", text=f"Function '{func_name}' removed successfully.")] # <-- Success message
                    else:
                        # Raise error to be caught by the main handler
                        raise RuntimeError(f"Function '{func_name}' could not be removed (function_remove returned False). Check logs.")

            elif name == "_function_add":
                # Add empty function
                func_name = args.get("name")
                if not func_name:
                    raise ValueError("Missing required parameter: name")

                logger.debug(f"---> Calling built-in: function_add for '{func_name}'") # <-- ADD THIS LINE

                # Check if function already exists
                function_path = os.path.join(FUNCTIONS_DIR, f"{func_name}.py")
                if os.path.exists(function_path):
                    # Function already exists - inform the client rather than raise error
                    # Create annotation dict for 'function already exists' error
                    error_message = f"Function '{func_name}' already exists"
                    error_annotations = {
                        "tool_error": {"tool_name": name, "message": error_message}
                    }
                    result_raw = [TextContent(type="text", text=error_message, annotations=error_annotations)]
                else: # <-- ADD else block
                    # Create empty function (stub) using dynamic_function_manager.function_add (raise error on failure)
                    added = await self.function_manager.function_add(func_name)
                    if added:
                        try:
                            await self._notify_tool_list_changed(change_type="added", tool_name=func_name) # Pass params
                        except Exception as e:
                            logger.error(f"Error sending tool notification after adding {func_name}: {str(e)}")
                        result_raw = [TextContent(type="text", text=f"Empty function '{func_name}' created successfully.")] # <-- Success message
                    else:
                        # Raise error to be caught by the main handler
                        raise RuntimeError(f"Function '{func_name}' could not be added (function_add returned False). Check logs.")

            elif name == "_server_get":
                svc_name = args.get("name")
                if not svc_name:
                    raise ValueError("Missing required parameter: name")
                logger.debug(f"---> Calling built-in: server_get for '{svc_name}'")
                result_raw = await self.server_manager.server_get(svc_name)
            elif name == "_server_add":
                svc_name = args.get("name")
                if not svc_name:
                    raise ValueError("Missing required parameter: 'name' must be a string")
                logger.debug(f"---> Calling built-in: server_add for '{svc_name}'")
                # Now server_add only requires the name parameter and creates a template
                success = await self.server_manager.server_add(svc_name)
                if success:
                    result_raw = [TextContent(type="text", text=f"MCP '{svc_name}' added successfully.")]
                else:
                    result_raw = [TextContent(type="text", text=f"Failed to add MCP '{svc_name}'.")]
            elif name == "_server_remove":
                svc_name = args.get("name")
                if not svc_name:
                    raise ValueError("Missing required parameter: name")
                logger.debug(f"---> Calling built-in: server_remove for '{svc_name}'")
                success = await self.server_manager.server_remove(svc_name)
                result_raw = [TextContent(type="text", text=f"Server '{svc_name}' removed successfully.")] if success else [TextContent(type="text", text=f"Failed to remove server '{svc_name}'.")]
            elif name == "_server_set":
                logger.debug(f"---> Calling built-in: server_set with args: {args!r}")
                result_raw = await self.server_manager.server_set(args, self)
            elif name == "_server_validate":
                svc_name = args.get("name")
                if not svc_name:
                    raise ValueError("Missing required parameter: name")
                logger.debug(f"---> Calling built-in: server_validate for '{svc_name}'")
                result_raw = await self.server_manager.server_validate(svc_name)
            elif name == "_server_start":
                logger.debug(f"---> Calling built-in: server_start with args: {args!r}")
                result_raw = await self.server_manager.server_start(args, self)
            elif name == "_server_stop":
                logger.debug(f"---> Calling built-in: server_stop with args: {args!r}")
                result_raw = await self.server_manager.server_stop(args, self)
            elif name == "_server_get_tools":
                server_name = args.get('name')
                if not server_name or not isinstance(server_name, str):
                     raise ValueError("Missing or invalid 'name' argument for _server_get_tools")
                result_raw = await self.server_manager.get_server_tools(server_name) # Pass only the name string
                # Convert Tool objects to dictionaries for JSON serialization
                if result_raw and isinstance(result_raw, list):
                    result_raw = [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "inputSchema": tool.inputSchema
                        }
                        for tool in result_raw
                    ]
            # Handle MCP tool calls
            elif '.' in name or ' ' in name: # <<< UPDATED Condition

                # --- Handle MCP tool call ---

                logger.info(f"🌐 MCP TOOL CALL: {name}")
                # Split on the first occurrence of '.' or ' '
                server_alias, tool_name_on_server = re.split('[. ]', name, 1) # <<< UPDATED Splitting
                logger.debug(f"Parsed: Server Alias='{server_alias}', Remote Tool='{tool_name_on_server}'")

                # Check if MCP server config exists and is running
                if server_alias not in self._server_configs: # Access instance variable
                       raise ValueError(f"Unknown server alias: '{server_alias}'")
                if not await self.server_manager.is_server_running(server_alias):
                    raise ValueError(f"Server '{server_alias}' is not running.")

                # Get the session for the target server
                task_info = self.server_manager.active_server_tasks.get(server_alias)
                if not task_info:
                     # This shouldn't happen if the check above passed, but safety first
                    raise ValueError(f"Could not find task info for running server '{server_alias}'.")

                session = task_info.get('session')
                ready_event = task_info.get('ready_event')

                if not session and ready_event:
                    session_ready_timeout = 5.0 # Allow a bit more time for proxy calls
                    logger.debug(f"Session for '{server_alias}' not immediately ready for proxy call. Waiting up to {session_ready_timeout}s...")
                    try:
                        await asyncio.wait_for(ready_event.wait(), timeout=session_ready_timeout)
                        session = task_info.get('session') # Re-fetch session after wait
                        if not session:
                            raise ValueError(f"Server '{server_alias}' session not available even after waiting.")
                        logger.debug(f"Session for '{server_alias}' became ready.")
                    except asyncio.TimeoutError:
                         raise ValueError(f"Timeout waiting for server '{server_alias}' session to become ready for proxy call.")
                elif not session:
                     # Session not available and no ready_event to wait for
                     raise ValueError(f"Server '{server_alias}' is running but its session is not available and cannot wait (no ready_event).")

                # Proxy the call using the retrieved session
                try:
                    logger.info(f"🌐 PROXYING tool call '{tool_name_on_server}' to server '{server_alias}' with args: {args}")
                    # Use the standard request timeout defined elsewhere
                    proxy_response: CallToolResult = await asyncio.wait_for(
                        session.call_tool(tool_name_on_server, args or {}),
                        timeout=SERVER_REQUEST_TIMEOUT
                    )
                    logger.info(f"✅ PROXY response received from '{server_alias}'")
                    logger.debug(f"Raw Proxy Response: {proxy_response}") # proxy_response is CallToolResult

                    # Directly use the content from the proxied result if it's a list.
                    # The isError flag from proxy_response is implicitly handled by the caller
                    # receiving our server's CallToolResult (which we don't explicitly build here,
                    # the framework does it based on the returned content list and exceptions).
                    if proxy_response.content and isinstance(proxy_response.content, list):
                         final_result = proxy_response.content
                         logger.debug(f"Using content list directly from proxied response: {final_result}")
                         logger.debug(f"<--- _execute_tool RETURNING proxied result directly.")
                         return final_result
                    else:
                        # Fallback: If content is missing or not a list - THIS is an error in proxy process.
                        error_message = f"Proxied server '{server_alias}' returned unexpected content format (expected list): {proxy_response.content}"
                        logger.error(error_message)
                        # Raise an exception, which the framework will turn into an error response
                        raise ValueError(error_message)

                except McpError as mcp_err:
                    logger.error(f"❌ MCPError proxying tool call '{name}' to '{server_alias}': {mcp_err}", exc_info=True)
                    # Format the MCPError into a user-friendly error message
                    error_message = f"Error calling '{tool_name_on_server}' on server '{server_alias}': {mcp_err.message} (Code: {mcp_err.code})"
                    raise ValueError(error_message) from mcp_err
                except asyncio.TimeoutError:
                    logger.error(f"❌ Timeout proxying tool call '{name}' to '{server_alias}'.")
                    raise ValueError(f"Timeout calling '{tool_name_on_server}' on server '{server_alias}'.")
                except Exception as proxy_err:
                    logger.error(f"❌ Unexpected error proxying tool call '{name}' to '{server_alias}': {proxy_err}", exc_info=True)
                    raise ValueError(f"Unexpected error calling '{tool_name_on_server}' on server '{server_alias}': {proxy_err}") from proxy_err

            elif not name.startswith('_'): # <--- CHANGED HERE

                # --- Handle Local Dynamic Function Call ---
                logger.info(f"🔧 CALLING LOCAL DYNAMIC FUNCTION: {name}")

                # warn if cached load error
                if name in _runtime_errors:
                     load_error_info = _runtime_errors[name]
                     logger.warning(f"❌ Function '{name}' has a cached load error: {load_error_info['error']}")
                     # Maybe return a specific error message here instead of raising ValueError?
                     # Creating an error TextContent for consistency

                # Check if function exists
                function_path = os.path.join(FUNCTIONS_DIR, f"{name}.py")
                if not os.path.exists(function_path):
                    raise ValueError(f"Function '{name}' not found")

                # Call the dynamic function
                try:
                    # Dynamic functions are directly handled by name matching
                    # Add detailed logging to show exactly what we're receiving from the cloud
                    logger.info(f"RECEIVED FROM CLOUD: Tool: '{name}', Raw Args: {args!r}, Type: {type(args)}")
                    if user:
                        logger.info(f"Call made by user: {user}")
                    logger.debug(f"---> Calling dynamic: function_call for '{name}' with args: {args} and client_id: {client_id} and request_id: {request_id}, user: {user}") # Log args and client_id separately
                    # Pass arguments, client_id and user distinctly
                    result_raw = await self.function_manager.function_call(name=name, client_id=client_id, request_id=request_id, user=user, args=args)
                    logger.debug(f"<--- Dynamic function '{name}' RAW result: {result_raw} (type: {type(result_raw)})")
                except Exception as e:
                    logger.error(f"❌ Error during dynamic function call '{name}': {str(e)}", exc_info=True)
                    raise ValueError(f"Error executing function '{name}': {str(e)}") from e


            else:
                # Handle unknown tool names starting with '_', or if no branch matched
                logger.error(f"❓ Unknown or unhandled tool name: {name}")
                raise ValueError(f"Unknown or unhandled tool name: {name}")

            # ---> ADDED: Process raw result into final_result format (List[TextContent])
            if isinstance(result_raw, str):
                final_result = [TextContent(type="text", text=result_raw)]
            elif isinstance(result_raw, dict):
                # Serialize dict to JSON string and wrap in TextContent
                logger.debug(f"<--- Serializing dict result to JSON string for tool '{name}'.")
                import json
                try:
                    json_string = json.dumps(result_raw)
                    result_content = [TextContent(type="text", text=json_string, annotations={"sourceType": "json"})] # <--- CHANGE HERE
                except TypeError as e:
                    logger.error(f"Error serializing dictionary result to JSON for tool '{name}': {e}")
                    result_content = [TextContent(type="error", text=f"Error serializing result: {e}")]
                final_result = result_content
            elif isinstance(result_raw, list) and all(isinstance(item, TextContent) for item in result_raw):
                final_result = result_raw
            elif result_raw is None: # Handle cases where built-ins might not have set result_raw (e.g., error occurred before assignment)
                logger.warning(f"⚠️ result_raw was None for tool '{name}'. This might indicate an unhandled path or early error.")
                final_result = [] # Assign a default empty list
            else:
                # Convert any other result to string
                import json
                try:
                    result_str = json.dumps(result_raw)
                    final_result = [TextContent(type="text", text=result_str, annotations={"sourceType": "json"})] # <--- CHANGE HERE
                except TypeError:
                    result_str = str(result_raw) # Fallback to plain string conversion
                    logger.warning(f"⚠️ Tool '{name}' returned non-standard type {type(result_raw)}. Converting to string: {result_str}")
                    final_result = [TextContent(type="text", text=result_str)]

            # ---> ADDED: Log final result before returning
            logger.debug(f"<--- _execute_tool RETURNING final result: {final_result!r}") # <-- ADD THIS LINE
            return final_result

        except Exception as e:
            logger.error(f"❌ Error in _execute_tool for '{name}': {str(e)}", exc_info=True)
            # Return a generic error message as TextContent with tool_error annotation dict
            error_message = f"Error executing tool '{name}': {str(e)}"
            error_annotations = {
                "tool_error": {"tool_name": name, "message": str(e)} # Use original exception message
            }
            final_result = [TextContent(type="text", text=error_message, annotations=error_annotations)]
            logger.debug(f"<--- _execute_tool RETURNING error result: {final_result!r}") # <-- ADD THIS LINE
            return final_result

    async def _notify_tool_list_changed(self, change_type: str, tool_name: str):
        """Send a 'notifications/tools/list_changed' notification with details to all connected clients."""
        logger.info(f"🔔 Notifying clients about tool list change ({change_type}: {tool_name})...")
        notification_params = {
            "changeType": change_type,
            "toolName": tool_name
        }
        notification = {
            "jsonrpc": "2.0",
            "method": "notifications/tools/list_changed", # MCP spec format
            "params": notification_params # Include details in params
        }
        notification_json = json.dumps(notification)

        # Access global connection tracking
        global client_connections

        if not client_connections:
            logger.debug("No clients connected, skipping tool list change notification.")
            return

        # Deduplicate cloud connections to ensure we only notify each unique cloud connection once
        unique_connections = {}
        for client_id, info in client_connections.items():
            client_type = info.get("type")
            connection = info.get("connection")

            if client_type == "cloud":
                # Use connection object ID as key to deduplicate
                conn_id = id(connection)
                if conn_id not in unique_connections:
                    unique_connections[conn_id] = client_id
                    logger.debug(f"Using client_id {client_id} for cloud connection {conn_id}")
            else:
                # For non-cloud connections, just use the client_id directly
                unique_connections[client_id] = client_id

        logger.debug(f"Found {len(unique_connections)} unique connections from {len(client_connections)} client entries")

        # Iterate through the deduplicated client IDs
        client_ids = list(unique_connections.values())
        for client_id in client_ids:
            if client_id not in client_connections: # Check if client disconnected during iteration
                continue

            client_info = client_connections[client_id]
            client_type = client_info.get("type")
            connection = client_info.get("connection")

            if not connection:
                logger.warning(f"No connection object found for client {client_id}, skipping notification.")
                continue

            try:
                if client_type == "websocket":
                    await connection.send_text(notification_json)
                    logger.debug(f"📢 Sent notifications/tools/list_changed to WebSocket client: {client_id}")
                elif client_type == "cloud" and connection.is_connected:
                    await connection.send_message('mcp_notification', notification)
                    logger.debug(f"☁️ Sent notifications/tools/list_changed to Cloud client: {client_id}")
                else:
                    logger.warning(f"Unknown or disconnected client type for {client_id}, skipping notification.")

            except Exception as e:
                logger.warning(f"Failed to send notifications/tools/list_changed notification to client {client_id}: {e}")
                # Consider removing the client connection if sending fails repeatedly?


async def get_all_tools_for_response(server: 'DynamicAdditionServer', caller_context: str) -> List[Dict[str, Any]]:
    """
    Fetches all tools from the server and prepares them as dictionaries for a JSON response.
    """
    logger.debug(f"Helper: Calling _get_tools_list for all tools from {caller_context}")
    raw_tool_list: List[Tool] = await server._get_tools_list(caller_context=caller_context)
    tools_dict_list: List[Dict[str, Any]] = []
    for tool in raw_tool_list:
        try:
            # Debug log to see server tools and their annotations BEFORE serialization
            if hasattr(tool, 'annotations') and isinstance(tool.annotations, dict) and tool.annotations.get('type') == 'server':
                logger.debug(f"🔍 SERIALIZING SERVER TOOL '{tool.name}' with annotations: {tool.annotations}")
                # Check if started_at is in annotations
                if 'started_at' in tool.annotations:
                    logger.debug(f"✅ Found started_at in annotations for '{tool.name}': {tool.annotations['started_at']}")

            # Ensure model_dump is called correctly for each tool
            tool_dict = tool.model_dump(mode='json') # Use mode='json' for better serialization

            # Debug log for server tools AFTER serialization
            if tool_dict.get('annotations', {}).get('type') == 'server':
                logger.debug(f"🔍 SERIALIZED SERVER TOOL '{tool_dict.get('name')}' to dict: {tool_dict}")
                # Check if started_at is in annotations
                if 'started_at' in tool_dict.get('annotations', {}):
                    logger.debug(f"✅ Started_at preserved in serialized tool dict for '{tool_dict.get('name')}': {tool_dict['annotations']['started_at']}")

                    # If started_at is in annotations but not at the top level, add it to the top level
                    started_at_val = tool_dict['annotations']['started_at']
                    if 'started_at' not in tool_dict:
                        logger.debug(f"🔎 Adding started_at to TOP LEVEL for '{tool_dict.get('name')}': {started_at_val}")
                        tool_dict['started_at'] = started_at_val
                else:
                    logger.debug(f"❌ Started_at MISSING in serialized tool dict for '{tool_dict.get('name')}'!")

            tools_dict_list.append(tool_dict)
        except Exception as e:
            error_msg = f"Error serializing tool: {e}"
            logger.error(f"❌ Error dumping tool model '{tool.name}' to dict: {e}", exc_info=True)

            # Create a placeholder that preserves as much original tool info as possible
            tool_error_dict = {
                "name": tool.name if hasattr(tool, 'name') else "unknown_tool",
                "description": tool.description if hasattr(tool, 'description') else "",
                "parameters": tool.parameters.model_dump() if hasattr(tool, 'parameters') and hasattr(tool.parameters, 'model_dump') else {"type": "object", "properties": {}},
            }

            # Add error information to annotations without changing other fields
            annotations = {}
            if hasattr(tool, 'annotations') and tool.annotations:
                # Try to preserve original annotations if possible
                try:
                    if hasattr(tool.annotations, 'model_dump'):
                        annotations = tool.annotations.model_dump()
                    elif isinstance(tool.annotations, dict):
                        annotations = tool.annotations.copy()
                except Exception:
                    pass  # If we can't get original annotations, use empty dict

            # Add error info to annotations
            annotations["error"] = True
            annotations["error_message"] = error_msg
            tool_error_dict["annotations"] = annotations
            tools_dict_list.append(tool_error_dict)  # Include the error info instead of skipping
    logger.debug(f"Helper: Prepared {len(tools_dict_list)} tool dictionaries.")
    return tools_dict_list

async def get_filtered_tools_for_response(server: 'DynamicAdditionServer', caller_context: str) -> List[Dict[str, Any]]:
    """
    Fetches tools, filters out server-type tools, and prepares them for a JSON response.
    """
    logger.debug(f"Helper: Calling get_all_tools_for_response for filtering from {caller_context}")
    all_tools_dict_list = await get_all_tools_for_response(server, caller_context)

    filtered_tools_dict_list: List[Dict[str, Any]] = []
    filtered_out_names: List[str] = []

    for tool_dict in all_tools_dict_list:
        # Check annotations safely
        annotations = tool_dict.get('annotations')
        if isinstance(annotations, dict) and annotations.get('type') == 'server':
            filtered_out_names.append(tool_dict.get('name', '<Unnamed Tool>'))
        else:
            filtered_tools_dict_list.append(tool_dict)

    if filtered_out_names:
        logger.info(f"🐾 Helper: Filtered out {len(filtered_out_names)} server-type tools from list requested by {caller_context}: {', '.join(filtered_out_names)}")

    logger.debug(f"Helper: Returning {len(filtered_tools_dict_list)} filtered tool dictionaries.")
    return filtered_tools_dict_list





# ServiceClient class to manage the connection to the cloud server via Socket.IO
class ServiceClient:
    """Socket.IO client for connecting to the cloud server

    This class implements a Socket.IO CLIENT to connect TO the cloud server.
    Socket.IO is different from standard WebSockets - it adds features like:
    - Automatic reconnection
    - Fallback to long polling when WebSockets aren't available
    - Namespaces for multiplexing
    - Authentication handling

    While server.py acts as a WebSocket SERVER for the node-mcp-client,
    it must act as a Socket.IO CLIENT to connect to the cloud server.

    Manages the Socket.IO connection to the cloud server's service namespace.
    """
    def __init__(self, server_url: str, namespace: str, email: str, api_key: str, serviceName: str, mcp_server: Server, port: int):
        self.server_url = server_url
        self.namespace = namespace
        self.email = email
        self.api_key = api_key
        self.serviceName = serviceName
        self.mcp_server = mcp_server
        self.server_port = port # Store the server's listening port
        self.sio = None
        self.retry_count = 0
        self.connection_task = None
        self.is_connected = False
        self.connection_active = True
        # Store creation time for stable client ID
        self._creation_time = int(time.time())

    async def connect(self):
        """Establish a Socket.IO connection to the cloud server"""
        logger.info(f"☁️ CONNECTING TO CLOUD SERVER: {self.server_url} (namespace: {self.namespace})")
        self.connection_active = True
        self.connection_task = asyncio.create_task(self._maintain_connection())
        return self.connection_task

    async def _maintain_connection(self):
        """Maintains the connection to the cloud server with retries"""
        while self.connection_active and not is_shutting_down:
            logger.info("☁️ Starting _maintain_connection loop iteration") # DEBUG ADDED
            try:
                # Create a new Socket.IO client instance
                self.sio = socketio.AsyncClient()

                # Register event handlers
                self._register_event_handlers()

                logger.info(f"☁️ Attempting connection to cloud server (attempt {self.retry_count + 1})")

                # Connect with authentication data including hostname
                import socket
                hostname = socket.gethostname()
                await self.sio.connect(
                    self.server_url,
                    namespaces=[self.namespace],
                    transports=['websocket'],  # Prefer websocket
                    auth={
                        "email": self.email,
                        "apiKey": self.api_key,
                        "serviceName": self.serviceName,
                        "hostname": hostname,
                        "port": self.server_port # Send the stored port
                    },
                    retry=False # We handle retries manually with backoff
                )

                # Wait for disconnection
                await self.sio.wait()
                logger.info("☁️ Socket.IO connection closed (wait() returned)") # DEBUG ADDED

            except Exception as e:
                if is_shutting_down:
                    logger.info("☁️ Shutting down, stopping cloud connection attempts")
                    break

                self.is_connected = False
                self.sio = None

                detailed_error_info_message = "Check DEBUG logs for a full traceback."
                error_message = str(e)

                if CLOUD_CONNECTION_MAX_RETRIES is not None and self.retry_count >= CLOUD_CONNECTION_MAX_RETRIES:
                    logger.error(f"❌ FAILED TO CONNECT TO CLOUD SERVER AFTER {self.retry_count} ATTEMPTS: {error_message}. {detailed_error_info_message}")
                    logger.error("❌ GIVING UP ON CLOUD CONNECTION!")
                    break

                self.retry_count += 1

                # Enhanced error message logic
                if error_message == "One or more namespaces failed to connect":
                    if self.email and self.api_key:
                        specific_error_log = f"Authentication failed. Please make sure credentials are valid (email: {self.email}, API key, service name: {self.serviceName})"
                    else:
                        specific_error_log = f"Namespace '{self.namespace}' failed to connect. This could be due to server-side issues, incorrect namespace configuration, or missing authentication details if required by the server."
                elif isinstance(e, socketio.exceptions.ConnectionError):
                    specific_error_log = f"{error_message}. This often indicates a network issue, the cloud server at {self.server_url} being down, or a firewall blocking the connection."
                else:
                    specific_error_log = f"{error_message}. An unexpected error occurred during connection. {detailed_error_info_message}"

                logger.error(f"❌ EGAD!! ATLANTIS CLOUD SERVER CONNECTION ERROR (attempt {self.retry_count}): {specific_error_log}")

                # Print a more detailed stack trace for debugging
                import traceback
                logger.debug(f"Traceback: {traceback.format_exc()}")

                # Calculate exponential backoff delay with jitter
                backoff_delay = CLOUD_CONNECTION_RETRY_SECONDS * (1.2 ** self.retry_count)
                jitter = random.uniform(0, 1) # Add random jitter (0-1 seconds)
                actual_delay = min(backoff_delay + jitter, CLOUD_CONNECTION_MAX_BACKOFF_SECONDS)

                # Wait before retrying
                logger.info(f"☁️ RETRYING CLOUD CONNECTION IN {actual_delay:.2f} SECONDS...")
                await asyncio.sleep(actual_delay)

        logger.info("☁️ Cloud connection giving up")

    def print_ascii_art(self, filepath):
        """
        Reads an ASCII art file and prints its content to the console.

        Args:
            filepath (str): The path to the ASCII art file.
        """
        try:
            with open(filepath, 'r') as file:
                ascii_art = file.read()
                print(ascii_art)
        except FileNotFoundError:
            print(f"Error: The file '{filepath}' was not found.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def _register_event_handlers(self):
        """Register Socket.IO event handlers"""
        if not self.sio:
            return

        # Connection established event
        @self.sio.event(namespace=self.namespace)
        async def connect(): # Ensure handler is async
            self.is_connected = True
            self.retry_count = 0  # Reset retry counter on successful connection

            self.print_ascii_art("../kitty.txt")

            logger.info("") # Blank line before
            logger.info(f"{BOLD}{CYAN}=================================================={RESET}")
            logger.info(f"{BOLD}{BRIGHT_WHITE}🚀✨🎉 CONNECTED TO ATLANTIS CLOUD SERVER! 🎉✨🚀{RESET}")
            logger.info(f"{BOLD}{CYAN}=================================================={RESET}")
            logger.info("") # Blank line after

            # --- ADDED: Register this connection with the MCP server ---
            cloud_sid = self.sio.sid if self.sio else 'unknown_sid' # Get Socket.IO session ID if available
            connection_id = f"service_{cloud_sid}"
            self.mcp_server.service_connections[connection_id] = {
                "type": "service",
                "connection": self.sio,
                "id": connection_id
            }
            logger.info(f"✅ Registered cloud service connection: {connection_id}")
            # -------------------------------------------------------------

            # Get the list of tools to log them
            tools_list = await self.mcp_server._get_tools_list(caller_context="_handle_connect_cloud")
            tool_names = [tool.name for tool in tools_list]
            logger.info(f"📊 REGISTERING {len(tools_list)} TOOLS WITH CLOUD: {', '.join(tool_names)}")

            # Emit the client event upon successful connection
            await self.send_message('client', {'status': 'connected'})

        # Connection error event
        @self.sio.event(namespace=self.namespace)
        def connect_error(data):
            logger.error(f"❌ DETAILED CLOUD CONNECTION FAILURE INFO: {data}") # Made log distinct

        # Disconnection event
        @self.sio.event(namespace=self.namespace)
        async def disconnect(): # Ensure handler is async
            logger.warning("⚠️ DISCONNECTED FROM CLOUD SERVER! (disconnect event)") # DEBUG ADDED
            self.is_connected = False
            logger.info("☁️ DISCONNECTED FROM CLOUD SERVER!") # DEBUG ADDED

            # --- ADDED: Unregister this connection from the MCP server ---
            cloud_sid = self.sio.sid if self.sio else 'unknown_sid'
            connection_id = f"service_{cloud_sid}"
            removed = self.mcp_server.service_connections.pop(connection_id, None)
            if removed:
                logger.info(f"✅ Removed cloud service connection: {connection_id}")
            else:
                logger.warning(f"⚠️ Tried to remove dead cloud service connection: {connection_id}")
            # --------------------------------------------------------------

            # If disconnection was not intentional (e.g., server shutdown), try reconnecting
            if self.connection_active and not is_shutting_down:
                logger.info("☁️ Attempting to reconnect to cloud server...")
                # The _maintain_connection loop will handle the retry logic
            else:
                logger.info("☁️ Disconnection was expected or shutdown initiated, not reconnecting.")
        # Service message event
        @self.sio.event(namespace=self.namespace)
        async def service_message(data):
            logger.debug(f"☁️ RAW RECEIVED SERVICE MESSAGE: {data}")
            # Check if this is an MCP JSON-RPC request
            if isinstance(data, dict) and 'jsonrpc' in data and 'method' in data:
                # This is an MCP JSON-RPC request
                response = await self._process_mcp_request(data)
                if response:
                    await self.send_message('mcp_response', response)
            else:
                # Ignore non-JSON-RPC messages or log a warning
                logger.warning(f"⚠️ Received non-JSON-RPC message, ignoring: {data}")

    async def _process_mcp_request(self, request: dict) -> Union[dict, None]:
        """Process an MCP JSON-RPC request from the cloud server by manually routing
        to the appropriate logic method in the DynamicAdditionServer.
        """
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        # Use persistent cloud client ID for this connection
        client_id = f"cloud_{self._creation_time}_{id(self)}"
        logger.debug(f"Created persistent cloud client ID: {client_id}")

        logger.info(f"☁️ Processing MCP request via manual routing: {method} (ID: {request_id})")

        response = {
            "jsonrpc": "2.0",
            "id": request_id
        }

        try:

            if method == "tools/list":
                logger.info(f"🧰 Processing 'tools/list' request via helper")
                filtered_tools_dict_list = await get_filtered_tools_for_response(self.mcp_server, caller_context="process_mcp_request_websocket")
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"tools": filtered_tools_dict_list}
                }
                logger.debug(f"📦 Prepared tools/list response (ID: {request_id}) with {len(filtered_tools_dict_list)} tools.")
                return response
            elif method == "tools/list_all": # Handling for list_all in direct connections
                # get all tools including internal
                # get servers including those not running
                # Call the core logic method directly (pass client_id)
                logger.info(f"🧰 Processing 'tools/list_all' request via helper")
                all_tools_dict_list = await get_all_tools_for_response(self.mcp_server, caller_context="process_mcp_request_websocket")
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"tools": all_tools_dict_list}
                }
                return response

            elif method == "tools/call":
                tool_name = params.get("name")
                tool_args = params.get("arguments") # MCP spec uses 'arguments'

                if tool_name is None or tool_args is None:
                    response["error"] = {"code": -32602, "message": "Invalid params: missing tool name or arguments"}
                else:
                    logger.debug(f"☁️ Calling _execute_tool for: {tool_name}")
                    # Register this client connection
                    global client_connections
                    client_connections[client_id] = {"type": "cloud", "connection": self}

                    # Extract 'user' field if present at the top level of the request
                    user = request.get("user", None)
                    if user:
                        logger.debug(f"☁️ Request includes user field: {user}")

                    # Call the core logic method directly with client ID, request ID, and user
                    call_result_list = await self.mcp_server._execute_tool(name=tool_name, args=tool_args, client_id=client_id, request_id=request_id, user=user)
                    # The _execute_tool method ensures result is List[TextContent]
                    # Convert TextContent objects to dictionaries for JSON serialization using model_dump()
                    # IMPORTANT: We use "contents" (plural) key to match format between Python and Node servers
                    try:
                        # Manually convert TextContent objects to dictionaries
                        contents_list = []
                        for content in call_result_list:
                            try:
                                if hasattr(content, 'model_dump'):
                                    content_data = content.model_dump()
                                    contents_list.append(content_data)
                                else:
                                    # Handle non-pydantic objects
                                    logger.warning(f"⚠️ Non-pydantic content object: {type(content)}")
                                    if hasattr(content, 'to_dict'):
                                        contents_list.append(content.to_dict())
                                    else:
                                        # Fallback for simple objects
                                        contents_list.append({"type": "text", "text": str(content)})
                            except Exception as e:
                                logger.error(f"❌ Error serializing content result: {e}")
                                # Add simple text content as fallback
                                contents_list.append({"type": "text", "text": str(content)})

                        # Construct the response according to JSON-RPC 2.0 and MCP spec
                        response["result"] = {"contents": contents_list}
                    except Exception as e:
                        logger.error(f"❌ Error constructing final response: {e}")
                        response["error"] = {"code": -32000, "message": "Internal server error during response formatting"}

            else:
                # Unknown method
                response["error"] = {"code": -32601, "message": f"Method not found: {method}"}

            return response

        except Exception as e:
            logger.error(f"❌ ERROR PROCESSING MCP REQUEST: {str(e)}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            response["error"] = {"code": -32000, "message": str(e)}
            return response

    async def send_message(self, event: str, data: dict):
        """Send a message to the cloud server via a named event"""
        if not self.sio or not self.is_connected:
            logger.warning(f"⚠️ ATTEMPTED TO SEND {event} BUT NOT CONNECTED")
            return False

        try:
            # IMPORTANT: Always use the original event name for all messages
            # Cloud server expects 'mcp_notification' for client logs
            emit_event = event

            # Special logging for client logs to make debugging easier
            if event == 'mcp_notification' and isinstance(data, dict) and data.get('method') == 'notifications/message':
                logger.debug(f"☁️ SENDING CLIENT LOG via {emit_event}: {data.get('method')}")
            else:
                logger.debug(f"☁️ SENDING MESSAGE: {emit_event}")

            # Emit the event with the appropriate name
            await self.sio.emit(emit_event, data, namespace=self.namespace)

            return True
        except Exception as e:
            logger.error(f"❌ ERROR SENDING MESSAGE: {str(e)}")
            return False

    async def disconnect(self):
        """Disconnect from the cloud server"""
        logger.info("☁️ DISCONNECTING FROM CLOUD SERVER")
        self.connection_active = False
        if self.sio and self.is_connected:
            try:
                await self.sio.disconnect()
            except Exception as e:
                logger.warning(f"⚠️ ERROR DURING DISCONNECT: {str(e)}")
        if self.connection_task and not self.connection_task.done():
            self.connection_task.cancel()
            try:
                await self.connection_task
            except asyncio.CancelledError:
                pass
        logger.info("☁️ CLOUD SERVER CONNECTION CLOSED")

# Global collection to track active websocket connections
active_websockets = set()

# Global dictionary to track client connections by ID
client_connections = {}

# Global dictionary to store dynamically registered clients (loaded from file)
REGISTERED_CLIENTS: Dict[str, Dict[str, Any]] = {}
CLIENTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "registered_clients.json")

# --- ADDED Persistence Functions ---
def _load_registered_clients():
    """Load registered clients from the JSON file into memory."""
    global REGISTERED_CLIENTS
    if os.path.exists(CLIENTS_FILE):
        try:
            with open(CLIENTS_FILE, 'r') as f:
                REGISTERED_CLIENTS = json.load(f)
                logger.info(f"💾 Loaded {len(REGISTERED_CLIENTS)} registered clients from {CLIENTS_FILE}")
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"❌ Error loading {CLIENTS_FILE}: {e}. Starting with empty registrations.")
            REGISTERED_CLIENTS = {}
    else:
        logger.info(f"ℹ️ Client registration file not found ({CLIENTS_FILE}). Starting fresh.")
        REGISTERED_CLIENTS = {}

def _save_registered_clients():
    """Save the current registered clients dictionary to the JSON file."""
    try:
        with open(CLIENTS_FILE, 'w') as f:
            json.dump(REGISTERED_CLIENTS, f, indent=2) # Use indent for readability
            logger.debug(f"💾 Saved {len(REGISTERED_CLIENTS)} registered clients to {CLIENTS_FILE}") # DEBUG level
    except IOError as e:
        logger.error(f"❌ Error saving registered clients to {CLIENTS_FILE}: {e}")
# --- End Persistence Functions ---

# --- ADDED BACK Global MCP Server Instantiation ---
mcp_server = DynamicAdditionServer()

# Custom WebSocket handler for the MCP server
async def handle_websocket(websocket: WebSocket):
    # Accept the WebSocket connection with MCP subprotocol
    await websocket.accept(subprotocol="mcp")

    # Generate a unique client ID based on address
    client_id = f"ws_{websocket.client.host}_{id(websocket)}"

    # Track this websocket connection both globally and by ID
    global active_websockets, client_connections
    active_websockets.add(websocket)
    client_connections[client_id] = {"type": "websocket", "connection": websocket}
    connection_count = len(active_websockets)

    logger.info(f"🔌 New WebSocket connection established from {websocket.client.host} (ID: {client_id}, Active: {connection_count})")

    try:
        # Message loop
        while True:
            # Wait for a message from the client
            message = await websocket.receive_text()

            try:
                # Parse the message as JSON
                request = json.loads(message)
                logger.debug(f"📥 Received: {request}")

                # Process the request using our MCP server (include client_id)
                response = await process_mcp_request(mcp_server, request, client_id)

                # Send the response back to the client
                #logger.debug(f"📤 Sending: {response}")
                logger.debug(f"📤 Sending response")
                await websocket.send_text(json.dumps(response))

            except json.JSONDecodeError:
                logger.error(f"🚫 Invalid JSON received: {message}")
                await websocket.send_text(json.dumps({"error": "Invalid JSON"}))
            except Exception as e:
                logger.error(f"🚫 Error processing request: {e}")
                await websocket.send_text(json.dumps({"error": str(e)}))

    except WebSocketDisconnect as e:
        logger.info(f"☑️ WebSocket client disconnected normally: code={e.code}, reason={e.reason}")
    except Exception as e:
        logger.error(f"🛑 WebSocket error: {e}")
    finally:
        # Remove this connection from all tracking
        active_websockets.discard(websocket)

        # Find and remove from client_connections
        to_remove = []
        for cid, info in client_connections.items():
            if info.get("type") == "websocket" and info.get("connection") is websocket:
                to_remove.append(cid)
        for cid in to_remove:
            client_connections.pop(cid, None)

        connection_count = len(active_websockets)
        logger.info(f"👋 WebSocket connection closed with {websocket.client.host} (Active: {connection_count})")

# Process MCP request and generate response
async def process_mcp_request(server, request, client_id=None):
    """Process an MCP request and return a response

    Args:
        server: The MCP server instance
        request: The request to process
        client_id: Optional ID of the requesting client for tracking
    """

    logger.info(f"🚀 Processing MCP request")

    if "id" not in request:
        return {"error": "Missing request ID"}

    req_id = request.get("id")
    method = request.get("method")
    params = request.get("params", {})

    # Store client_id in thread-local storage or other request context
    # This allows tools called during this request to know who's calling
    global current_request_client_id
    current_request_client_id = client_id

    # Route the request to the appropriate handler
    try:
        if method == "initialize":
            # Process initialize request
            logger.info(f"🚀 Processing 'initialize' request with params: {params}")
            result = await server.initialize(params)
            logger.info(f"✅ Successfully processed 'initialize' request")
            # Return empty object per MCP protocol spec
            return {"jsonrpc": "2.0", "id": req_id, "result": result}
        elif method == "tools/list":
            logger.info(f"🧰 Processing 'tools/list' request via helper")
            filtered_tools_dict_list = await get_filtered_tools_for_response(server, caller_context="process_mcp_request_websocket")
            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": filtered_tools_dict_list}
            }
            logger.debug(f"📦 Prepared tools/list response (ID: {req_id}) with {len(filtered_tools_dict_list)} tools.")
            return response
        elif method == "tools/list_all": # Handling for list_all in direct connections
            # get all tools including internal
            # get servers including those not running
            # Call the core logic method directly (pass client_id)
            logger.info(f"🧰 Processing 'tools/list_all' request via helper")
            all_tools_dict_list = await get_all_tools_for_response(server, caller_context="process_mcp_request_websocket")
            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": all_tools_dict_list}
            }
            return response

        elif method == "prompts/list":
            result = await server._get_prompts_list()
            return {"jsonrpc": "2.0", "id": req_id, "result": {"prompts": result}}
        elif method == "resources/list":
            result = await server._get_resources_list()
            return {"jsonrpc": "2.0", "id": req_id, "result": {"resources": result}}
        elif method == "tools/call":
            name = params.get("name")
            args = params.get("arguments", {}) # MCP spec uses 'arguments'
            user = params.get("user", None) # Extract the 'user' field that tells us who is making the call
            logger.info(f"🔧 Processing 'tools/call' for tool '{name}' with args: {args}")

            # Log the tool name and arguments for debugging
            logger.debug(f"Tool name: '{name}', Arguments: {json.dumps(args, default=str)}")
            if user:
                logger.debug(f"Call made by user: {user}")

            # Execute the tool - pass the user field
            result = await server._execute_tool(name=name, args=args, client_id=client_id, request_id=req_id, user=user)
            logger.info(f"🎯 Tool '{name}' execution completed with {len(result)} content items")

            # Debug what kind of result we got
            logger.debug(f"Result type: {type(result)}, Items: {len(result)}")
            for i, item in enumerate(result):
                logger.debug(f"  Item {i}: {type(item).__name__}")

            # Format response according to JSON-RPC 2.0 spec for MCP
            try:
                # Manually convert TextContent objects to dictionaries
                contents_list = []
                for content in result:
                    try:
                        if hasattr(content, 'model_dump'):
                            content_data = content.model_dump()
                            contents_list.append(content_data)
                        else:
                            # Handle non-pydantic objects
                            logger.warning(f"⚠️ Non-pydantic content object: {type(content)}")
                            if hasattr(content, 'to_dict'):
                                contents_list.append(content.to_dict())
                            else:
                                # Fallback for simple objects
                                contents_list.append({"type": "text", "text": str(content)})
                    except Exception as e:
                        logger.error(f"❌ Error serializing content result: {e}")
                        # Add simple text content as fallback
                        contents_list.append({"type": "text", "text": str(content)})

                # Construct the response according to JSON-RPC 2.0 and MCP spec
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": contents_list
                    }
                }
                logger.debug(f"📦 Formatted response: {json.dumps(response, default=str)[:200]}...")
                return response
            except Exception as e:
                logger.error(f"💥 Error formatting tool call response: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                return {"jsonrpc": "2.0", "id": req_id, "error": f"Error formatting tool call response: {e}"}
        else:
            logger.warning(f"⚠️ Unknown method requested: {method}")
            return {"jsonrpc": "2.0", "id": req_id, "error": f"Unknown method: {method}"}
    except Exception as e:

        import traceback
        logger.error(f"🚫 Error processing request '{method}': {e}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return {"jsonrpc": "2.0", "id": req_id, "error": f"Error processing request: {e}"}

# Set up the Starlette application with routes
async def handle_registration(request: Request) -> JSONResponse:
    """Handle dynamic client registration requests (POST /register).

    Expects JSON body with at least 'client_name' and 'redirect_uris'.
    Generates client_id and client_secret.
    Persists client data to file.
    Based on RFC 7591.
    """
    try:
        client_metadata = await request.json()
        logger.info(f"🔑 Received registration request: {client_metadata}")

        # --- Enhanced Validation ---
        client_name = client_metadata.get("client_name")
        redirect_uris = client_metadata.get("redirect_uris")

        if not client_name:
            return JSONResponse({"error": "invalid_client_metadata", "error_description": "Missing 'client_name'"}, status_code=400)
        if not redirect_uris or not isinstance(redirect_uris, list) or not all(isinstance(uri, str) for uri in redirect_uris):
            return JSONResponse({"error": "invalid_redirect_uri", "error_description": "'redirect_uris' must be a non-empty array of strings"}, status_code=400)
        # Add more validation for other DCR params (grant_types, response_types, scope etc.) if needed

        # --- Client Creation ---
        client_id = str(uuid.uuid4())
        issued_at = int(datetime.datetime.utcnow().timestamp()) # Use timestamp
        client_secret = secrets.token_urlsafe(32) # Generate client secret

        # --- Store Client Details ---
        # Store all provided valid metadata
        registered_data = {
            "client_id": client_id,
            "client_name": client_name,
            "redirect_uris": redirect_uris,
            "client_id_issued_at": issued_at,
            "client_secret": client_secret, # Store client secret
            "client_secret_expires_at": 0, # 0 means never expires, or set a timestamp
            # Store other validated DCR fields here (grant_types, response_types, scope etc.)
            # "token_endpoint_auth_method": client_metadata.get("token_endpoint_auth_method", "client_secret_basic") # Default or from request
        }
        REGISTERED_CLIENTS[client_id] = registered_data
        _save_registered_clients() # Save after modification

        logger.info(f"✅ Registered new client: ID={client_id}, Name='{client_name}', URIs={redirect_uris}")


        # Return the registered client metadata (INCLUDING secret for M2M simplicity for now)
        response_data = registered_data.copy()
        # Consider *not* returning the secret in production for higher security

        return JSONResponse(response_data, status_code=201) # 201 Created

    except json.JSONDecodeError:
        logger.error("❌ Registration failed: Invalid JSON in request body")
        return JSONResponse({"error": "invalid_client_metadata", "error_description": "Invalid JSON format"}, status_code=400)
    except Exception as e:
        logger.error(f"❌ Registration failed: {str(e)}", exc_info=True)
        return JSONResponse({"error": "internal_server_error", "error_description": "Internal server error during registration"}, status_code=500)

app = Starlette(
    routes=[
        WebSocketRoute("/mcp", endpoint=handle_websocket),
        Route("/register", endpoint=handle_registration, methods=["POST"])
    ]
)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="MCP WebSocket Server")
    parser.add_argument("--host", default=HOST, help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=PORT, help="Port to bind the server to")
    parser.add_argument("--cloud-host", default=CLOUD_SERVER_HOST, help="Cloud server host to connect to")
    parser.add_argument("--cloud-port", type=int, default=CLOUD_SERVER_PORT, help="Cloud server port to connect to")
    parser.add_argument("--cloud-namespace", default=CLOUD_SERVICE_NAMESPACE, help="Cloud server Socket.IO namespace")
    parser.add_argument("--email", help="Service email for cloud authentication")
    parser.add_argument("--api-key", help="Service API key for cloud authentication")
    parser.add_argument("--service-name", help="Desired service name")
    parser.add_argument("--no-cloud", action="store_true", help="Disable cloud server connection")
    args = parser.parse_args()

    # Update host and port from command line arguments
    HOST = args.host
    PORT = args.port

    _load_registered_clients() # Load clients at startup

    # Update cloud server settings if provided
    if args.cloud_host != CLOUD_SERVER_HOST or args.cloud_port != CLOUD_SERVER_PORT:
        CLOUD_SERVER_HOST = args.cloud_host
        CLOUD_SERVER_PORT = args.cloud_port
        CLOUD_SERVER_URL = f"{CLOUD_SERVER_HOST}:{CLOUD_SERVER_PORT}"

    # Set up the event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Initialize the MCP server
    logger.info(f"{BRIGHT_WHITE}🔧 === CALLING SERVER INITIALIZE FROM MAIN ==={RESET}")
    loop.run_until_complete(mcp_server.initialize())

    # Ensure dynamic directories exist
    os.makedirs(FUNCTIONS_DIR, exist_ok=True)
    logger.info(f"📁 Dynamic functions directory: {FUNCTIONS_DIR}")
    os.makedirs(SERVERS_DIR, exist_ok=True)
    logger.info(f"📁 Dynamic servers directory: {SERVERS_DIR}")

    # Start the file watcher
    event_handler = DynamicConfigEventHandler(mcp_server, loop)
    observer = Observer()
    observer.schedule(event_handler, FUNCTIONS_DIR, recursive=False) # Don't watch subdirs
    observer.schedule(event_handler, SERVERS_DIR, recursive=False)
    observer.start()
    logger.info(f"👁️ Watching for changes in {FUNCTIONS_DIR} and {SERVERS_DIR}...")

    # Start the Uvicorn server
    config = uvicorn.Config(app, host=HOST, port=PORT, log_level="warning")
    server = uvicorn.Server(config)

    # Create tasks for the server and cloud connection
    server_task = loop.create_task(server.serve())
    cloud_task = None

    try:
        # Start the server
        logger.info(f"🌟 STARTING LOCAL MCP WEBSOCKET SERVER AT ws://{HOST}:{PORT}/mcp")

        # Connect to cloud server if enabled
        if not args.no_cloud:
            if not args.email or not args.api_key or not args.service_name:
                logger.error("❌ CLOUD SERVER CONNECTION REQUIRES EMAIL, API KEY AND SERVICE NAME")
                logger.error("❌ Use --email and --api_key to specify credentials, --service-name to specify desired service name")
                logger.info("☁️ CLOUD SERVER CONNECTION DISABLED")
            else:
                logger.info(f"{PINK}☁️ CLOUD SERVER CONNECTION ENABLED: {CLOUD_SERVER_URL}{RESET}")
                # Create the cloud connection with the provided credentials
                cloud_connection = ServiceClient(
                    server_url=CLOUD_SERVER_URL,
                    namespace=CLOUD_SERVICE_NAMESPACE,
                    email=args.email,
                    api_key=args.api_key,
                    serviceName=args.service_name,
                    mcp_server=mcp_server,
                    port=PORT # Pass the listening port
                )
                cloud_task = loop.create_task(cloud_connection.connect())
        else:
            logger.info("☁️ CLOUD SERVER CONNECTION DISABLED")

        # Run the event loop until the server is interrupted
        loop.run_until_complete(server_task)
    except KeyboardInterrupt:
        logger.info("👋 RECEIVED KEYBOARD INTERRUPT")
    finally:
        # Cancel any pending tasks
        if cloud_task and not cloud_task.done():
            cloud_task.cancel()
        server_task.cancel()

        # Clean up the loop
        pending = asyncio.all_tasks(loop=loop)
        for task in pending:
            task.cancel()

        logger.info("🧹 CLEANING UP TASKS")
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()
        pid_manager.remove_pid_file() # Ensure PID file is removed on exit
        logger.info("👋 SERVER SHUTDOWN COMPLETE")
