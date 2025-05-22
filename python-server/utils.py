#!/usr/bin/env python3
import os
import psutil
import logging
import json
import asyncio
from state import logger

# ANSI escape codes for colors
PINK = "\x1b[95m"  # Added Pink
RESET = "\x1b[0m"



"""
Utility functions available for dynamic functions to use.
Provides easy access to client-side logging and other shared functionality.
"""

import logging
from typing import Any

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
    message_type: str = "text"
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
            logger.info(f"📋 CLIENT LOG DIAGNOSTICS (AWAITABLE): Routing to client_id={client_id_for_routing}, request_id={request_id}")
            logger.info(f"📋 SERVER INSTANCE TYPE (AWAITABLE): {type(_server_instance).__name__}")

            # Await the server call to send the log/command and get a response
            if logger_name is None:
                logger_name = "dynamic_function"

            # Pass request_id, client_id_for_routing, AND seq_num to the server's method
            # Pass all parameters including message_type to the server's method
            # ASSUMPTION: _server_instance.send_client_log is now an async def method
            # that returns the actual result/status from the client/Atlantis.
            result = await _server_instance.send_client_log(
                level,
                message,
                logger_name,
                request_id,
                client_id_for_routing,
                seq_num,
                entry_point_name,
                message_type
            )

            logger.info(f"📋 CLIENT LOG/COMMAND (AWAITABLE) SENT, result: {result}")
            return result # Return the result from the server call
        except Exception as e:
            logger.error(f"❌ Error in awaitable client_log: {e}")
            logger.error(f"❌ Exception details: {type(e).__name__}: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            raise # Re-raise the exception so the caller (atlantis.py) can handle it
    else:
        logger.warning("Cannot send client log/command: server instance not set")
        return None # Or raise an error, depending on desired behavior


# --- JSON Formatting Utility --- #

def format_json_log(data: dict) -> str:
    """Formats a Python dictionary into a pretty-printed JSON string for logging."""
    try:
        return json.dumps(data, indent=2, default=str) # Added default=str to handle non-serializable types gracefully
    except Exception as e:
        logger.error(f"❌ Error formatting JSON for logging: {e}")
        return str(data) # Fallback to string representation
