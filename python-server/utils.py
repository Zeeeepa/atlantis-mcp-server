#!/usr/bin/env python3
import os
import psutil
import logging
import json
import asyncio
from state import logger
from typing import Any, Optional

# ANSI escape codes for colors
PINK = "\x1b[95m"  # Added Pink
RESET = "\x1b[0m"


"""
Utility functions available for dynamic functions to use.
Provides easy access to client-side logging and other shared functionality.
"""


# Empty function for filename cleaning - placeholder for future implementation
def clean_filename(name: str) -> str:
    """
    Cleans/sanitizes a filename for filesystem usage.
    This is a placeholder function that will be implemented later.

    Args:
        name: The filename to clean

    Returns:
        Cleaned filename safe for filesystem usage
    """
    return name

# Global server reference to be set at startup
_server_instance = None

def set_server_instance(server):
    """Set the server instance for dynamic functions to use."""
    global _server_instance
    _server_instance = server
    logger.debug("Server instance set in utils module")

def get_server_instance():
    global _server_instance
    return _server_instance

async def client_log(
    message: Any,
    level: str = "info",
    logger_name: str = None,
    request_id: str = None,
    client_id_for_routing: str = None,
    seq_num: int = None,
    entry_point_name: str = None,
    message_type: str = "text",
    stream_id: Optional[str] = None
    ):
    """
    Send a log message to the client.

    IMPORTANT: This is the LOW-LEVEL implementation of client logging.
    Dynamic functions should NOT call this directly! Instead, use atlantis.client_log,
    which is a context-aware wrapper that automatically provides all the necessary
    context variables (request_id, client_id, etc.) to this function.

    atlantis.client_log -> utils.client_log -> server.send_client_log

    This function can be imported and called from dynamic functions to send
    logs directly to the client using MCP notifications, but atlantis.client_log
    is preferred in most cases.

    Args:
        message: The message to log (can be a string or structured data)
        level: Log level ("debug", "info", "warning", "error")
        logger_name: Optional name to identify the logger source
        request_id: Optional ID of the original request that triggered this log
        client_id_for_routing: Optional client identifier to route the log message
        seq_num: Optional sequence number for client-side ordering
        entry_point_name: Name of the top-level function called by the request.
        message_type: Type of message content ("text", "json", "image/png", etc.). Default is "text"
    """
    # Log locally first (always using INFO level for local display)
    seq_prefix = f"(Seq: {seq_num}) " if seq_num is not None else ""
    log_prefix = f"{PINK}CLIENT LOG [{level.upper()}] {seq_prefix}"
    log_suffix = f"(Client: {client_id_for_routing}, Req: {request_id}, Entry: {entry_point_name}, Logger: {logger_name}): {message}{RESET}"
    logger.info(f"{log_prefix}{log_suffix}") # Add seq_num to local log too

    # Send to client if server is available
    if _server_instance is not None:
        try:
            # Enhanced logging for diagnostics
            logger.debug(f"📋 CLIENT LOG DIAGNOSTICS (AWAITABLE): Routing to client_id={client_id_for_routing}, request_id={request_id}")
            logger.debug(f"📋 SERVER INSTANCE TYPE (AWAITABLE): {type(_server_instance).__name__}")

            # Await the server call to send the log/command and get a response
            if logger_name is None:
                logger_name = "dynamic_function"

            # Pass request_id, client_id_for_routing, AND seq_num to the server's method
            # Pass all parameters including message_type to the server's method
            # Create a task to send the client log without awaiting its completion here.
            # This makes utils.client_log effectively fire-and-forget from the caller's perspective.
            async def send_log_task():
                try:
                    task_result = await _server_instance.send_client_log(
                        level,
                        message,
                        logger_name,
                        request_id,
                        client_id_for_routing,
                        seq_num,
                        entry_point_name,
                        message_type,
                        stream_id
                    )
                    logger.debug(f"📋 CLIENT LOG/COMMAND (TASK) SENT, result: {task_result}")
                except Exception as task_e:
                    logger.error(f"❌ Error in send_log_task: {task_e}")
                    logger.error(f"❌ Exception details: {type(task_e).__name__}: {task_e}")
                    import traceback
                    logger.error(f"❌ Traceback: {traceback.format_exc()}")

            asyncio.create_task(send_log_task())
            logger.debug(f"📋 CLIENT LOG/COMMAND TASK CREATED for client_id={client_id_for_routing}, request_id={request_id}, seq_num={seq_num}")
            return None # Return immediately, indicating task creation
        except Exception as e:
            logger.error(f"❌ Error in awaitable client_log: {e}")
            logger.error(f"❌ Exception details: {type(e).__name__}: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            raise # Re-raise the exception so the caller (atlantis.py) can handle it
    else:
        logger.warning("Cannot send client log/command: server instance not set")
        return None # Or raise an error, depending on desired behavior

async def execute_client_command_awaitable(
    client_id_for_routing: str,
    request_id: str, # Original MCP request ID for client context
    command: str, # The command string for the client
    command_data: Optional[Any] = None, # Optional data for the command
    seq_num: Optional[int] = None, # Sequence number for client-side ordering
    entry_point_name: Optional[str] = None # Entry point name for logging
    ) -> Any:
    """
    Sends a command to a specific client via the server and waits for a dedicated response.
    This is a low-level utility intended to be called by atlantis.client_command for awaitable operations.
    It calls the server's 'send_awaitable_client_command' method.

    Args:
        client_id_for_routing: The ID of the client to send the command to.
        request_id: The original MCP request ID, for client-side context.
        command: The command identifier string.
        command_data: Optional data payload for the command.
        seq_num: Optional sequence number for client-side ordering.
        entry_point_name: Optional name of the entry point function for logging.

    Returns:
        The result from the client's command execution, as returned by the server.

    Raises:
        McpError: If the server call fails, client response times out, or client returns an error.
        RuntimeError: If the server instance is not available or doesn't support awaitable commands.
    """
    global _server_instance
    if _server_instance is None:
        logger.error("❌ Cannot execute awaitable client command: server instance not set.")
        raise RuntimeError("Server instance not available for awaitable command.")

    if not hasattr(_server_instance, 'send_awaitable_client_command'):
        logger.error("❌ Server instance does not have 'send_awaitable_client_command' method. This is required for true awaitable commands.")
        raise RuntimeError("Server instance is outdated or incorrect for awaitable commands.")

    try:
        logger.info(f"🚀 Utils: Relaying dedicated awaitable command '{command}' to server for client {client_id_for_routing}, seq_num={seq_num}")
        # This specifically calls the method designed for awaitable command-response cycles
        result = await _server_instance.send_awaitable_client_command(
            client_id_for_routing=client_id_for_routing,
            request_id=request_id,
            command=command,
            command_data=command_data,
            seq_num=seq_num,
            entry_point_name=entry_point_name
        )
        logger.info(f"✅ Utils: Received result for dedicated awaitable command '{command}' from server.")
        return result
    except Exception as e:
        # Errors (including McpError for timeouts/client errors from server's send_awaitable_client_command) will propagate
        logger.error(f"❌ Utils: Error relaying dedicated awaitable command '{command}' or receiving result: {type(e).__name__} - {e}")
        raise # Re-raise the exception for atlantis.py to handle or propagate further

# --- JSON Formatting Utility --- #

def format_json_log(data: dict) -> str:
    """Formats a Python dictionary into a pretty-printed JSON string for logging."""
    try:
        return json.dumps(data, indent=2, default=str) # Added default=str to handle non-serializable types gracefully
    except Exception as e:
        logger.error(f"❌ Error formatting JSON for logging: {e}")
        return str(data) # Fallback to string representation
