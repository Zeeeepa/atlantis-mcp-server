"""
Manages the lifecycle of dynamic MCP servers (configs) by launching them
as background tasks using the MCP Python SDK's stdio transport.
Stores each server config as a JSON file under SERVERS_DIR.
"""
import os
import asyncio
import json
import logging
import pathlib
import shutil
import datetime
from typing import Any, Dict, Optional, List, Tuple, Union

from mcp import ClientSession, StdioServerParameters, stdio_client
from mcp.types import TextContent, Tool, ListToolsResult, ListToolsRequest

from state import (
    SERVERS_DIR,
    logger,
    SERVER_REQUEST_TIMEOUT
)

# --- Tracking for Active Server Tasks (Moved from server_manager) ---
# Stores {'server_name': {'task': asyncio.Task, 'config': Dict, 'shutdown_event': asyncio.Event, 'session': Optional[ClientSession], 'ready_event': asyncio.Event}}
ACTIVE_SERVER_TASKS: dict[str, dict] = {}


class DynamicServerManager:

    def __init__(self, servers_dir: str):
        self.servers_dir = servers_dir
        self.old_dir = os.path.join(servers_dir, 'OLD')

        # Ensure directories exist
        os.makedirs(self.servers_dir, exist_ok=True)
        os.makedirs(self.old_dir, exist_ok=True)




        self.active_server_tasks = ACTIVE_SERVER_TASKS
        self.server_start_times = {}
        self._server_load_errors = {}

        logger.info(f"Dynamic Server Manager initialized with servers dir: {servers_dir}")

    # --- 1. File Save/Load Methods ---

    def extract_server_name(self, server_json_content: str) -> Optional[str]:
        try:
            # Try to parse the JSON content
            config = json.loads(server_json_content)

            # Check if it's in the modern format with mcpServers
            if isinstance(config, dict) and 'mcpServers' in config:
                # Get the first key under mcpServers
                if isinstance(config['mcpServers'], dict) and len(config['mcpServers']) > 0:
                    return next(iter(config['mcpServers'].keys()))

            # Check if it's in the legacy format (top-level key)
            elif isinstance(config, dict) and len(config) > 0:
                # Return the first top-level key
                return next(iter(config.keys()))

            return None
        finally:
            pass

    async def _fs_save_server(self, name: str, content: str) -> Optional[str]:

        safe_name = f"{name}.json"
        file_path = os.path.join(self.servers_dir, safe_name)
        logger.debug(f"---> _fs_save_server: Attempting to save '{name}' to path: {file_path}")
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"💾 Saved server config for '{name}' to {file_path}")
            return file_path
        finally:
            pass

    async def _fs_load_server(self, name: str) -> Optional[str]:
        safe_name = f"{name}.json"
        file_path = os.path.join(self.servers_dir, safe_name)
        if not os.path.exists(file_path):
            logger.info(f"⚠️ _fs_load_server: Config file not found for '{name}' at {file_path}")
            self._server_load_errors.pop(name, None)  # Clear potential old error if file is gone
            return None
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()
            self._server_load_errors.pop(name, None) # Clear any previous load error for this name if read succeeds
            return raw_content
        except IOError as e:
            logger.error(f"❌ _fs_load_server: IOError reading server config '{name}' from {file_path}: {e}")
            self._server_load_errors[name] = f"IOError: {e}"
            return None

    async def _write_server_error_log(self, name: str, error_msg: str, raw_content: Optional[str] = None) -> None:
        log_filename = f"{name}_error.log"
        log_path = os.path.join(self.servers_dir, log_filename)
        try:
            with open(log_path, 'w', encoding='utf-8') as f:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"=== Error Log for '{name}' @ {timestamp} ===\n\n")
                f.write(f"{error_msg}\n\n")
                if raw_content:
                    f.write("=== Raw Content ===\n\n")
                    f.write(raw_content)
            logger.debug(f"📝 Error log written to {log_path}")
        except Exception as e:
            logger.error(f"❌ Failed to write error log for '{name}': {e}")

    # --- 2. Config CRUD Methods ---

    async def server_add(self, name: str) -> bool:
        # Check if server already exists
        existing = await self._fs_load_server(name)
        if existing is not None:
            logger.warning(f"⚠️ Server '{name}' already exists, not adding.")
            return False

        # Create a template config based on openweather example
        template_obj = {
            "mcpServers": {
                name: {
                    "command": "uvx",  # Default command, can be edited later
                    "args": [
                        "--from",
                        "atlantis-mcp-template",  # Placeholder package name
                        f"start-{name}-server",  # Create standard start command based on name
                        "--api-key",
                        "<your api key here>"  # Placeholder for API key
                    ]
                }
            }
        }

        # Save the new template config
        logger.info(f"🆕 Creating new server config template for '{name}'")
        template_config = json.dumps(template_obj, indent=4) # indent for pretty printing, optional
        result = await self._fs_save_server(name, template_config)
        return result is not None

    async def server_remove(self, name: str) -> bool:
        # Check if server is running and stop it first
        if name in self.active_server_tasks:
            logger.info(f"🛑 Stopping running server '{name}' before removal...")
            try:
                # Call server_stop with proper arguments
                await self.server_stop({"name": name}, None)
                logger.info(f"🔔 Successfully stopped server '{name}' before removal")
            except Exception as e:
                logger.error(f"❌ Failed to stop server '{name}' during removal: {e}")
                # Continue with removal anyway

        # Remove the config file
        file_path = os.path.join(self.servers_dir, f"{name}.json")
        if not os.path.exists(file_path):
            logger.warning(f"⚠️ Server config '{name}' not found for removal.")
            return False

        try:
            # Backup the config before deletion
            backup_dir = self.old_dir
            os.makedirs(backup_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(backup_dir, f"{timestamp}_{name}.json")
            shutil.copy2(file_path, backup_path)
            logger.info(f"📦 Backed up '{name}' config to {backup_path}")

            # Delete the original file
            os.remove(file_path)
            logger.info(f"🗑️ Removed server config '{name}'")

            # Clean up any cached errors
            self._server_load_errors.pop(name, None)

            return True
        except Exception as e:
            logger.error(f"❌ Failed to remove server '{name}': {e}")
            return False

    async def server_get(self, name: str) -> Optional[str]:
        # No validation, just return whatever _fs_load_server gives us
        return await self._fs_load_server(name)

    # Note: We're using the existing server_start method instead of a custom helper

    async def get_server_tools(self, name: str) -> List[Tool]:
        if not name or not isinstance(name, str):
            raise ValueError(f"Invalid server name: {name}")

        # Detailed logging for debugging
        logger.debug(f"get_server_tools: Checking for running server '{name}'")

        # Check if the server is already running (use existing session)
        if name in self.active_server_tasks:
            task_info = self.active_server_tasks[name]
            session = task_info.get('session')

            # If we have an existing session, try to use it
            if session is not None:
                logger.debug(f"get_server_tools: Found existing session for '{name}', attempting to use it")
                try:
                    # Check if session is usable by attempting a list_tools operation
                    # Use a timeout to prevent hanging if the session is stale
                    tools_result = await asyncio.wait_for(session.list_tools(), timeout=SERVER_REQUEST_TIMEOUT)

                    # Update last successful use timestamp
                    self.active_server_tasks[name]['last_used'] = datetime.datetime.now(datetime.timezone.utc)
                    logger.info(f"Successfully fetched {len(tools_result.tools)} tools from running server '{name}' using existing session")
                    return tools_result.tools
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout using existing session for '{name}', session may be stale")
                    # Continue to fetch tools via a new temporary connection
                except Exception as e:
                    logger.warning(f"Could not use existing session for '{name}': {e}")
                    # Continue to fetch tools via a new temporary connection
            else:
                logger.debug(f"Server '{name}' exists in active_server_tasks but session is not available yet")

                # Check if the server is still starting up
                if task_info.get('status') == 'starting' and 'ready_event' in task_info:
                    ready_event = task_info['ready_event']
                    logger.info(f"Server '{name}' is still starting, waiting for ready event")
                    try:
                        # Wait for the ready event with timeout
                        await asyncio.wait_for(ready_event.wait(), timeout=SERVER_REQUEST_TIMEOUT)

                        # If we're here, the ready event was set - check if a session is now available
                        if task_info.get('session') is not None:
                            # Try again with the new session
                            logger.debug(f"Server '{name}' is now ready, trying new session")
                            try:
                                tools_result = await asyncio.wait_for(task_info['session'].list_tools(), timeout=SERVER_REQUEST_TIMEOUT)
                                logger.info(f"Successfully fetched {len(tools_result.tools)} tools from newly started server '{name}'")
                                return tools_result.tools
                            except Exception as e:
                                logger.warning(f"Failed to use newly ready session for '{name}': {e}")
                                # Continue to temporary connection
                    except asyncio.TimeoutError:
                        logger.warning(f"Timed out waiting for server '{name}' to become ready")
                    # Continue to temporary connection if we couldn't use the session

        # Get the server config (for temporary connection or for starting the server)
        logger.debug(f"get_server_tools: Loading config for '{name}'")
        config_txt = await self.server_get(name)

        # convert config txt to json and store in full_config

        if 'mcpServers' not in full_config or name not in full_config['mcpServers']:
            raise ValueError(f"Invalid server config structure for '{name}'")

        server_config = full_config['mcpServers'][name]
        if 'command' not in server_config:
            raise ValueError(f"Missing 'command' in server config for '{name}'")

        # Auto-start the server if it's not already running
        if name not in self.active_server_tasks:
            logger.info(f"Server '{name}' is not running. Auto-starting it...")
            # Start the server and wait for it to be ready
            try:
                await self.server_start({'name': name}, None)  # Use the existing pattern with correct parameters
                logger.info(f"Successfully initiated auto-start for server '{name}'")
            except Exception as e:
                logger.error(f"Failed to auto-start server '{name}': {e}")
                raise RuntimeError(f"Failed to auto-start server '{name}': {e}")

            # Give it a moment to initialize fully
            logger.info(f"Waiting for server '{name}' to initialize...")
            await asyncio.sleep(1)  # Short delay to ensure initialization completes

            # Now retry getting the tools with the new session
            if name in self.active_server_tasks and self.active_server_tasks[name].get('session') is not None:
                session = self.active_server_tasks[name]['session']
                try:
                    # Use the new session
                    tools_result = await asyncio.wait_for(session.list_tools(), timeout=SERVER_REQUEST_TIMEOUT)
                    self.active_server_tasks[name]['last_used'] = datetime.datetime.now(datetime.timezone.utc)
                    logger.info(f"Successfully fetched {len(tools_result.tools)} tools from newly started server '{name}'")
                    return tools_result.tools
                except Exception as e:
                    logger.error(f"Failed to get tools from newly started server '{name}': {e}")
                    raise RuntimeError(f"Error fetching tools from newly started server '{name}': {e}")
            else:
                raise RuntimeError(f"Server '{name}' was started but session is not available")

        # If we got here, something unexpected happened
        raise RuntimeError(f"Unexpected error: Unable to get tools from server '{name}'")

    async def server_list(self) -> List[str]:
        try:
            servers_path = pathlib.Path(self.servers_dir)
            server_files = list(servers_path.glob('*.json'))
            server_names = [file.stem for file in server_files]
            return sorted(server_names)
        except Exception as e:
            logger.error(f"❌ Error listing servers: {e}")
            return []

    async def is_server_running(self, name: str) -> bool:
        if not name or not isinstance(name, str):
            return False

        # Check if the server exists in active_server_tasks with a valid task
        return (name in self.active_server_tasks and
                'task' in self.active_server_tasks[name] and
                not self.active_server_tasks[name]['task'].done())

    async def get_running_servers(self) -> List[str]:
        # Filter the active_server_tasks to only include properly running servers
        running_servers = []
        for name, task_info in self.active_server_tasks.items():
            # Check for basic task existence and not done
            if 'task' not in task_info or task_info['task'].done():
                logger.debug(f"Server '{name}' skipped: task missing or done")
                continue

            # Check for active session that's properly connected
            session = task_info.get('session')
            if session is None:
                logger.debug(f"Server '{name}' skipped: no session available")
                continue

            # For Python SDK ClientSession objects, we can't check 'connected' attribute
            # (as mentioned in your memory about mcp.ClientSession)
            # Instead, we'll assume if we have a session object, it's likely still usable
            # The actual tool fetching will validate this later and handle exceptions
            # This approach aligns with the recommendation to attempt a low-impact operation
            # instead of checking specific attributes
            logger.debug(f"Server '{name}' has a session - assuming it's valid")

            # This server passes all checks and is truly running
            logger.debug(f"Server '{name}' IS running: active task and session")
            running_servers.append(name)

        return running_servers

    async def server_set(self, name: str, content: str) -> List[TextContent]:
        # Save the content directly without any validation or processing
        result = await self._fs_save_server(name, content)

        # Simple success message - only return the TextContent list
        success_msg = f"Server '{name}' configuration saved."
        return [TextContent(type="text", text=success_msg)]

    async def server_validate(self, name: str) -> Dict[str, Any]:
        """
        Validates that the *saved* server config JSON has required keys.
        either returns JSON or raises exception
        """
        result = {
            'valid': False,
            'error': None
        }

        config_txt = await self.server_get(name)

        # convert config txt to json and store in config


        # Check for proper mcpServers structure
        if not isinstance(config, dict):
            result['error'] = f"Server config for '{name}' is not a valid dictionary"
            return result

        # Check if we have the mcpServers key which is the standard format
        if 'mcpServers' in config:
            # Modern format with mcpServers structure
            if not isinstance(config['mcpServers'], dict):
                result['error'] = f"'mcpServers' in config for '{name}' is not a dictionary"
                return result

            if name not in config['mcpServers']:
                result['error'] = f"Server '{name}' not found within 'mcpServers' key"
                return result

            server_config = config['mcpServers'][name]

            # Check required fields in the server config
            if not isinstance(server_config, dict):
                result['error'] = f"Server config for '{name}' is not a dictionary"
                return result

            if 'command' not in server_config:
                result['error'] = f"Missing required key 'command' in server config for '{name}'"
                return result

        else:
            # Legacy format (direct config)?
            logger.warning(f"⚠️ server_validate: Config for '{name}' does not use 'mcpServers' structure")

            # Check for command directly in the old format
            if 'command' not in config:
                result['error'] = f"Missing required key 'command' in server config for '{name}'"
                return result

        # Config is valid
        result['valid'] = True
        return result

    # --- 3. Background Task Methods ---

    async def _run_mcp_client_session(self, name: str, params: StdioServerParameters, shutdown_event: asyncio.Event) -> None:
        logger.info(f"[{name}] Starting MCP client session task...")
        session = None  # Define session here to ensure it's accessible in finally block if needed
        try:
            logger.debug(f"[{name}] Attempting stdio_client connection with params: {params}")
            # Simply initialize an empty output buffer
            if name in self.active_server_tasks:
                self.active_server_tasks[name]['output_buffer'] = []

            async with stdio_client(params) as (reader, writer):
                logger.debug(f"[{name}] Stdio connection established. Creating ClientSession...")
                # Create the session object without using context manager so we can keep it alive
                # This is crucial for connection reuse
                async with ClientSession(reader, writer) as session:

                    try:
                        # Initialize the session
                        logger.debug(f"[{name}] ClientSession created. Initializing session...")
                        await asyncio.wait_for(session.initialize(), timeout=15.0)  # Increased timeout for reliable init
                        logger.info(f"[{name}] MCP Session initialized successfully.")

                        # Update active_server_tasks *after* successful initialization
                        if name in self.active_server_tasks:
                            logger.debug(f"[{name}] Updating active_server_tasks with session and timestamp...")
                            self.active_server_tasks[name]['session'] = session
                            self.active_server_tasks[name]['started_at'] = datetime.datetime.now(datetime.timezone.utc)
                            self.active_server_tasks[name]['status'] = 'running'  # Mark as running
                            logger.info(f"[{name}] active_server_tasks updated. Current state for '{name}': {{'status': '{self.active_server_tasks[name].get('status')}', 'started_at': '{self.active_server_tasks[name].get('started_at')}', 'session_present': {self.active_server_tasks[name].get('session') is not None}}}")

                            # Pre-fetch tools list to validate the connection works
                            try:
                                tools_result = await session.list_tools()
                                logger.info(f"[{name}] Successfully fetched {len(tools_result.tools)} tools from running server")
                                logger.info(f"[{name}] Tools: {tools_result.tools}")
                                self.active_server_tasks[name]['tools_count'] = len(tools_result.tools)
                            except Exception as e:
                                logger.warning(f"[{name}] Failed to fetch tools list: {e}")

                            # Signal readiness *after* updating state
                            if 'ready_event' in self.active_server_tasks[name]:
                                self.active_server_tasks[name]['ready_event'].set()
                                logger.debug(f"[{name}] Ready event set.")
                            else:
                                logger.warning(f"[{name}] 'ready_event' not found in active_server_tasks entry.")
                        else:
                            logger.error(f"[{name}] Entry disappeared from active_server_tasks before session could be stored!")
                            # If the entry is gone, we probably can't signal readiness either. Shutdown might be needed.
                            return  # Exit if the task entry is gone

                        # --- Session is initialized and running ---
                        logger.debug(f"[{name}] Session active. Waiting for shutdown signal...")

                        # Keep the session alive until shutdown is requested
                        # Periodically check session is still responsive by calling a lightweight operation
                        while not shutdown_event.is_set():
                            try:
                                # Wait for shutdown event with timeout
                                # This allows us to periodically check if session is still responsive
                                shutdown_requested = await asyncio.wait_for(
                                    shutdown_event.wait(),
                                    timeout=30.0  # Check every 30 seconds
                                )
                                if shutdown_requested:
                                    break

                                # Optional: Check session is still responsive with lightweight ping
                                # Only do this for servers that have been running a while
                                if name in self.active_server_tasks and self.active_server_tasks[name].get('status') == 'running':
                                    started_at = self.active_server_tasks[name].get('started_at')
                                    now = datetime.datetime.now(datetime.timezone.utc)

                                    # Only ping if server has been running more than 2 minutes
                                    if started_at and (now - started_at).total_seconds() > 120:
                                        try:
                                            # Just list_tools as a lightweight ping
                                            await asyncio.wait_for(session.list_tools(), timeout=5.0)
                                            # No need to process the result, we're just checking if it responds
                                        except Exception as e:
                                            logger.warning(f"[{name}] Session health check failed: {e}")
                                            # Break the loop if session is no longer responsive
                                            # This will trigger cleanup in finally block
                                            break
                            except asyncio.TimeoutError:
                                # This is expected - it just means the shutdown_event.wait timed out
                                # We'll loop and check again
                                pass
                            except Exception as e:
                                # When we get any error in the session loop, capture it thoroughly
                                error_message = str(e)

                                # Detect package dependency errors in the error message
                                if "No solution found" in error_message or "not found in the package registry" in error_message:
                                    key_error = f"Package dependency error: {error_message.strip()}"
                                    logger.error(f"[{name}] 📦 PACKAGE ERROR DETECTED: {key_error}")
                                else:
                                    key_error = f"Startup error: {error_message.strip()}"
                                    logger.error(f"[{name}] 🚫 STARTUP ERROR: {key_error}")

                                # Save the error message in BOTH places for display in tools list
                                # This is the key - we must update both to ensure visibility
                                if name in self.active_server_tasks:
                                    self.active_server_tasks[name]['status'] = 'failed'
                                    self.active_server_tasks[name]['error'] = key_error
                                # Always update the persistent error storage
                                self._server_load_errors[name] = key_error

                                # Debug log the full error
                                if isinstance(e, BaseExceptionGroup):
                                    logger.error(f"[{name}] Error details:", exc_info=True)

                                # Close the session if it exists
                                if session is not None:
                                    try:
                                        await asyncio.wait_for(session.close(), timeout=5.0)
                                    except Exception as close_err:
                                        logger.debug(f"[{name}] Error closing session during cleanup: {close_err}")
                                break

                        logger.info(f"[{name}] Shutdown event received. Cleaning up session...")

                        # Close the session cleanly if we still have it
                        if session is not None:
                            try:
                                await asyncio.wait_for(session.close(), timeout=5.0)
                                logger.debug(f"[{name}] Session closed cleanly")
                            except Exception as e:
                                logger.warning(f"[{name}] Error closing session: {e}")

                    except asyncio.TimeoutError as timeout_err:
                        # Directly check for the uvx error - this is a common pattern in our server
                        common_errors = [
                            "No solution found when resolving tool dependencies",
                            "not found in the package registry",
                            "requirements are unsatisfiable"
                        ]

                        # Generic error message to start
                        error_msg = "Timeout occurred during session initialization"

                        # Look for common package dependency errors in logs
                        for line in self.active_server_tasks.get(name, {}).get('output_buffer', []):
                            for err_pattern in common_errors:
                                if err_pattern in line:
                                    # Found a more specific error message to use!
                                    error_msg = f"Package dependency error: {line.strip()}"
                                    logger.info(f"📦 [ERROR] Found package dependency error: {line}")
                                    break

                        logger.error(f"[{name}] {error_msg}")

                        if name in self.active_server_tasks:
                            # Mark server as failed
                            self.active_server_tasks[name]['status'] = 'failed'
                            # Set ready event to unblock anything waiting for this server
                            if 'ready_event' in self.active_server_tasks[name]:
                                self.active_server_tasks[name]['ready_event'].set()
                                logger.debug(f"[{name}] Ready event set (to unblock waiters) after initialization failure")
                            # Store the detailed error message
                            self.active_server_tasks[name]['error'] = error_msg
                            # Add to server load errors so it shows in tools list
                            self._server_load_errors[name] = error_msg

                            # Optional: Remove zombie tasks after a short delay to allow error inspection
                            async def _delayed_cleanup(server_name):
                                await asyncio.sleep(120)  # Keep failed entry for 2 minutes for diagnostics
                                if server_name in self.active_server_tasks and self.active_server_tasks[server_name].get('status') == 'failed':
                                    logger.info(f"[{server_name}] Removing failed server from active_server_tasks after delay")
                                    self.active_server_tasks.pop(server_name, None)

                            # Schedule cleanup
                            asyncio.create_task(_delayed_cleanup(name))

                        # Close session if it exists but failed to initialize
                        if session is not None:
                            try:
                                await session.close()
                            except Exception:
                                pass  # Ignore errors when closing an uninitialized session
                    except Exception as e:
                        logger.error(f"[{name}] Error during session.initialize() or operation: {e}")
                        if name in self.active_server_tasks:
                            self.active_server_tasks[name]['status'] = 'init_failed'
                            if 'ready_event' in self.active_server_tasks[name]:
                                self.active_server_tasks[name]['ready_event'].set()  # Signal completion (failure)

                        # Close session if it exists but failed
                        if session is not None:
                            try:
                                await session.close()
                            except Exception:
                                pass  # Ignore errors when closing a failed session

        except ConnectionRefusedError:
            logger.error(f"[{name}] Connection refused when starting stdio client process.")
            if name in self.active_server_tasks:
                self.active_server_tasks[name]['status'] = 'connection_refused'
                if 'ready_event' in self.active_server_tasks[name]:
                    self.active_server_tasks[name]['ready_event'].set()  # Signal completion (failure)

        except Exception as e:
            logger.error(f"[{name}] Unexpected error in client session: {e}", exc_info=True)
            if name in self.active_server_tasks:
                self.active_server_tasks[name]['status'] = 'error'
                if 'ready_event' in self.active_server_tasks[name]:
                    self.active_server_tasks[name]['ready_event'].set()  # Signal completion (failure)

        finally:
            # Clean up if needed (task cancellation, etc.)
            logger.info(f"[{name}] Client session task exiting.")

            # Force-clear session reference to help with garbage collection
            if session is not None:
                logger.debug(f"[{name}] Clearing session reference.")
                try:
                    await session.close()  # Ensure session is properly closed
                except Exception as e:
                    logger.debug(f"[{name}] Error closing session during cleanup: {e}")
                session = None

            # Remove from active tasks if still present
            if name in self.active_server_tasks:
                logger.debug(f"[{name}] Removing entry from active_server_tasks.")
                self.active_server_tasks.pop(name, None)
                self.server_start_times.pop(name, None)

    # --- 4. Server Start/Stop Methods ---

    async def server_start(self, args: Dict[str, Any], server) -> List[TextContent]:
        name = args.get('name')
        logger.debug(f"▶️ server_start: Entered with args: {args}")
        if not name or not isinstance(name, str):
            msg = "Missing or invalid parameter: 'name' must be str."
            logger.error(f"❌ server_start: {msg}")
            raise ValueError(msg)

        # Load the config using server_get (as in self_test)
        config_txt = await self.server_get(name)
        if not full_config:
            msg = f"Config file not found for server '{name}'."
            logger.error(f"❌ server_start: {msg}")
            raise FileNotFoundError(msg)


        if name in self.active_server_tasks:
            msg = f"Server '{name}' is already running."
            logger.warning(f"⚠️ server_start: {msg}")
            raise ValueError(msg)  # Or return a message indicating it's already running

        # Perform basic validation on the extracted config
        if not isinstance(server_config, dict) or 'command' not in server_config:
            msg = f"Invalid server config for '{name}': must be a dictionary with at least a 'command' key."
            logger.warning(f"⚠️ server_start: {msg}")
            raise ValueError(msg)

        validation_result = await self.server_validate(name)
        logger.debug(f"▶️ server_start: Validation result for '{name}': {validation_result}")
        if not validation_result.get('valid', False):
            error = validation_result.get('error', 'Unknown error')
            msg = f"Invalid config for '{name}': {error}"
            logger.error(f"❌ server_start: {msg}")
            raise ValueError(msg)

        try:
            # Prepare parameters for stdio_client
            params = StdioServerParameters(
                command=server_config['command'],
                args=server_config.get('args', []),
                env=server_config.get('env', {}),
                cwd=server_config.get('cwd', None)
            )
            logger.debug(f"▶️ server_start: Prepared StdioServerParameters for '{name}': {params}")
        except Exception as e:  # Catch potential Pydantic validation errors or KeyErrors
            msg = f"Failed to prepare start parameters for '{name}': {e}"
            logger.error(f"❌ server_start: {msg}", exc_info=True)
            raise ValueError(msg)

        # Start the background task
        logger.info(f"Attempting to start background task for server '{name}'...")
        logger.debug(f"▶️ server_start: Creating asyncio task for _run_mcp_client_session('{name}')...")
        shutdown_event = asyncio.Event()
        ready_event = asyncio.Event()  # Create the ready event

        # Start the MCP client session as an async task
        logger.debug(f"---> _add_server_task: Starting MCP client session for '{name}'")
        task = asyncio.create_task(self._run_mcp_client_session(name, params, shutdown_event))
        task_info = {
            'task': task,
            'config': full_config,
            'shutdown_event': shutdown_event,
            'ready_event': ready_event,
            'session': None,  # Will be populated once initialized
            'status': 'starting',  # Track status: starting -> running or failed
            'added_at': datetime.datetime.now(datetime.timezone.utc),
            'output_buffer': [],  # Store stdout/stderr from the process
            'error_buffer': []    # Store specific error messages
        }
        self.active_server_tasks[name] = task_info
        logger.info(f"MCP server '{name}' started")
        logger.debug(f"▶️ server_start: Task info recorded for '{name}'. Start time will be added upon session init.")

        # Return success
        return [TextContent(type='text', text=f"MCP service '{name}' started.")]

# ... (rest of the code remains the same)
    async def server_stop(self, args: Dict[str, Any], server) -> List[TextContent]:
        name = args.get('name')
        if not name or not isinstance(name, str):
            msg = "Missing or invalid parameter: 'name' must be str."
            logger.error(f"❌ server_stop: {msg}")
            raise ValueError(msg)

        if name not in self.active_server_tasks:
            msg = f"Server '{name}' is not running or not managed."
            logger.warning(f"⚠️ server_stop: {msg}")
            raise ValueError(msg)

        task_info = self.active_server_tasks.get(name)
        if not task_info or 'task' not in task_info:
            msg = f"Error stopping server '{name}': Inconsistent state."
            logger.error(f"❌ server_stop: {msg}")
            # Clean up if entry exists but is broken
            if name in self.active_server_tasks:
                del self.active_server_tasks[name]
            raise RuntimeError(msg)

        task = task_info['task']
        if task.done():
            logger.info(f"🧹 Task for server '{name}' was already finished. Cleaning up entry.")
            # Cleanup potentially missed by the finally block
            if name in self.active_server_tasks:
                del self.active_server_tasks[name]
            self.server_start_times.pop(name, None)  # Remove start time
            return [TextContent(type='text', text=f"Server '{name}' task was already finished.")]
        else:
            logger.info(f"Attempting to cancel task for server '{name}'...")

            # Set the shutdown event first to signal graceful shutdown
            if 'shutdown_event' in task_info and task_info['shutdown_event'] is not None:
                task_info['shutdown_event'].set()
                logger.debug(f"Shutdown event set for '{name}'")

            # Also cancel the task directly in case it's not responding to the event
            task.cancel()

            # Give cancellation a moment to potentially propagate
            try:
                # Wait briefly to see if cancellation completes quickly
                await asyncio.wait_for(task, timeout=1.0)
            except asyncio.CancelledError:
                logger.info(f"✅ Cancellation successful for server '{name}' task.")
            except asyncio.TimeoutError:
                logger.info(f"⏳ Task for server '{name}' cancellation initiated, may take time to fully stop.")
            except Exception as e:
                logger.error(f"❓ Unexpected error while waiting for cancellation of '{name}': {e}")

            # The finally block in _run_mcp_client_session should remove the entry.
            # Also clean up our tracking
            self.server_start_times.pop(name, None)  # Remove start time

            return [TextContent(type='text', text=f"MCP server '{name}' stopped")]

