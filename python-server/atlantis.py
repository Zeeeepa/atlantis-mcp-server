import contextvars
import inspect # Ensure inspect is imported
import asyncio # Added for Lock
from typing import Callable, Optional, Any, List # Added List
from utils import client_log as util_client_log # For client_log, client_image, client_html
from utils import execute_client_command_awaitable # For client_command
import uuid

# --- Context Variables ---
# client_log_func: Holds the partially bound client_log function for the current request
_client_log_var: contextvars.ContextVar[Optional[Callable]] = contextvars.ContextVar("_client_log_var", default=None)
# request_id: The unique ID of the current MCP request
_request_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("_request_id_var", default=None)
# client_id: The ID of the client that initiated the current request
_client_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("_client_id_var", default=None)
# log_seq_num: A counter for log messages within the current request context. Holds a list [int] to be mutable across tasks.
# It is initialized to [0] in server.py at the start of each request.
_log_seq_num_var: contextvars.ContextVar[Optional[List[int]]] = contextvars.ContextVar("_log_seq_num_var", default=None)
_seq_num_lock = asyncio.Lock() # Lock for synchronizing sequence number increments

# entry_point_name: The name of the top-level dynamic function called by the request
_entry_point_name_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("_entry_point_name_var", default=None)
# user: The user who made the call (from the 'user' field in tools/call requests)
_user_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("_user_var", default=None)

# --- Accessor Functions ---

async def client_log(message: Any, level: str = "INFO", message_type: str = "text"):
    """Sends a log message back to the requesting client for the current context.
    Includes a sequence number and automatically determines the calling function name.
    Also includes the original entry point function name.

    NOTE: This is a WRAPPER around the lower-level utils.client_log function.
    This function automatically captures context data (like request_id, caller name, etc.)
    and forwards it to utils.client_log. Dynamic functions should use THIS version,
    not utils.client_log directly.

    The message_type parameter specifies what kind of content is being sent:
    - "text" (default): A plain text message
    - "json": A JSON object or structured data
    - "image/*": Various image formats (e.g., "image/png", "image/jpeg")

    Calls the underlying log function directly; async dispatch is handled internally.
    """
    log_func = _client_log_var.get()
    if log_func:
        caller_name = "unknown_caller" # Default for immediate caller
        entry_point_name = _entry_point_name_var.get() or "unknown_entry_point" # Get entry point from context

        try:
            # --- Get immediate caller function name ---
            frame = inspect.currentframe()
            if frame and frame.f_back:
                caller_name = frame.f_back.f_code.co_name
            del frame
        except Exception as inspect_err:
            print(f"WARNING: Could not inspect caller frame for client_log: {inspect_err}")

        current_seq_to_send = -1 # Default to an invalid sequence number
        try:
            # Get current sequence number and increment it for the next call
            async with _seq_num_lock:
                seq_list_container = _log_seq_num_var.get()
                if seq_list_container is not None:
                    current_seq_to_send = seq_list_container[0]
                    seq_list_container[0] += 1
                else:
                    # This error print is outside the critical path of successful lock acquisition/increment
                    # but indicates a problem if _log_seq_num_var is None.
                    print(f"ERROR: client_log - _log_seq_num_var is None. Cannot get sequence number.")
                    # Handle error appropriately, perhaps by returning or raising an exception
                    # For now, it will proceed with current_seq_to_send = -1, which might be caught by the receiver or cause issues

            task = await util_client_log(
                client_id_for_routing=_client_id_var.get(),
                request_id=_request_id_var.get(),
                entry_point_name=entry_point_name,
                message_type=message_type,
                message=message,
                level=level,
                seq_num=current_seq_to_send # Pass the obtained sequence number
            )
            # task is the asyncio.Task returned by utils.client_log

            return None # Return None in either case
        except Exception as e:
            print(f"ERROR: Failed during async client_log call (after inspect): {e}")
            # Decide if to re-raise or return an error indicator
            raise
    else:
        print(f"WARNING: client_log called but no logger in context. Message: {message}")
        return None # Or raise an error

# --- Other Accessors ---
def get_request_id() -> Optional[str]:
    """Returns the request_id for the current context."""
    return _request_id_var.get()

def get_client_id() -> Optional[str]:
    """Returns the client_id for the current context."""
    return _client_id_var.get()

def get_user() -> Optional[str]:
    """Returns the user who made the call for the current context."""
    return _user_var.get()

# --- Setter Functions (primarily for internal use by dynamic_function_manager) ---

def set_context(client_log_func: Callable, request_id: str, client_id: str, entry_point_name: str, user: Optional[str] = None):
    """Sets all context variables and returns a tuple of their tokens for resetting."""
    client_log_token = _client_log_var.set(client_log_func)
    request_id_token = _request_id_var.set(request_id)
    client_id_token = _client_id_var.set(client_id)
    # Initialize _log_seq_num_var with a new list [0] for each call context
    log_seq_num_token = _log_seq_num_var.set([0])
    entry_point_token = _entry_point_name_var.set(entry_point_name)

    # Handle optional user context
    # Ensure _user_var is always set, even if to None, to get a valid token for reset_context
    actual_user = user if user is not None else None # Explicitly use None if user is not provided
    user_token = _user_var.set(actual_user)

    return (client_log_token, request_id_token, client_id_token, log_seq_num_token, entry_point_token, user_token)

def reset_context(tokens: tuple):
    """Resets the context variables using the provided tuple of tokens."""
    # Expected order: client_log, request_id, client_id, log_seq_num, entry_point, user
    if not isinstance(tokens, tuple) or len(tokens) != 6:
        print(f"ERROR: reset_context expected a tuple of 6 tokens, got {tokens}")
        # Add more robust error handling or logging as needed
        return

    # Unpack tokens
    client_log_token, request_id_token, client_id_token, log_seq_num_token, entry_point_token, user_token = tokens

    # Reset each context variable if its token is present (not strictly necessary with .set(None) giving a token)
    _client_log_var.reset(client_log_token)
    _request_id_var.reset(request_id_token)
    _client_id_var.reset(client_id_token)
    _log_seq_num_var.reset(log_seq_num_token)
    _entry_point_name_var.reset(entry_point_token)
    _user_var.reset(user_token) # user_token will be valid even if user was None


# --- Utility Functions ---

async def client_image(image_path: str, level: str = "INFO", image_format: str = "image/png"):
    """Sends an image back to the requesting client for the current context.
    This is a wrapper around client_log that automatically loads the image,
    converts it to base64, and sets the appropriate message type.

    Args:
        image_path: Path to the image file to send
        level: Log level (e.g., "INFO", "DEBUG")
        image_format: MIME type of the image (e.g., "image/png", "image/jpeg")

    Raises:
        FileNotFoundError: If the image file doesn't exist
        IOError: If there's an error reading the file
    """
    # Convert image to base64
    base64_data = image_to_base64(image_path)

    # Add 'base64:' prefix to the string
    prefixed_data = f"base64:{base64_data}"

    # Send to client_log with appropriate message_type
    # client_log is now async and returns a result
    result = await client_log(prefixed_data, level=level, message_type=image_format)
    return result

def image_to_base64(image_path: str) -> str:
    """Loads an image from the given file path and converts it to a base64 string.

    Args:
        image_path: The path to the image file to load

    Returns:
        A base64-encoded string representation of the image

    Raises:
        FileNotFoundError: If the image file doesn't exist
        IOError: If there's an error reading the file
    """
    import base64
    import os.path

    # Verify file exists to provide a helpful error
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        # Read the binary data from the file
        with open(image_path, "rb") as image_file:
            # Convert binary data to base64 encoded string
            encoded_string = base64.b64encode(image_file.read())
            # Return as UTF-8 string
            return encoded_string.decode('utf-8')
    except IOError as e:
        # Log the error for debugging
        print(f"Error converting image to base64: {e}")
        # Re-raise to allow caller to handle
        raise

async def stream_start() -> str:
    """Starts a new stream and returns a unique stream_id.
    Sends a 'stream_start' message to the client.
    """
    stream_id_to_send = str(uuid.uuid4())
    actual_client_id = _client_id_var.get() # Get the actual client ID from context

    request_id = _request_id_var.get()
    entry_point_name = _entry_point_name_var.get() or "unknown_entry_point"

    caller_name = "unknown_caller"
    try:
        frame = inspect.currentframe()
        if frame and frame.f_back:
            caller_name = frame.f_back.f_code.co_name
        del frame
    except Exception as inspect_err:
        print(f"WARNING: Could not inspect caller frame for stream_start: {inspect_err}")

    current_seq_to_send = -1 # Default to an invalid sequence number
    async with _seq_num_lock:
        seq_list_container = _log_seq_num_var.get()
        if seq_list_container is not None:
            current_seq_to_send = seq_list_container[0]
            seq_list_container[0] += 1
        else:
            print(f"ERROR: stream_start - _log_seq_num_var is None. Cannot get sequence number.")

    try:
        await util_client_log(
            seq_num=current_seq_to_send, # Pass the obtained sequence number
            message={"status": "started"},
            level="INFO",
            logger_name=caller_name,
            request_id=request_id,
            client_id_for_routing=actual_client_id, # Route using actual client_id
            entry_point_name=entry_point_name,
            message_type='stream_start',
            stream_id=stream_id_to_send # Pass the generated stream_id separately
        )
        return stream_id_to_send # Return the generated stream_id to the caller
    except Exception as e:
        print(f"ERROR: Failed during async stream_start call: {e}")
        raise

async def stream(message: str, stream_id_param: str):
    """Sends a stream message snippet back to the client using a provided stream_id.
    """
    actual_client_id = _client_id_var.get() # Get the actual client ID from context
    request_id = _request_id_var.get()
    entry_point_name = _entry_point_name_var.get() or "unknown_entry_point"

    caller_name = "unknown_caller"
    try:
        frame = inspect.currentframe()
        if frame and frame.f_back:
            caller_name = frame.f_back.f_code.co_name
        del frame
    except Exception as inspect_err:
        print(f"WARNING: Could not inspect caller frame for stream: {inspect_err}")

    current_seq_to_send = -1 # Default to an invalid sequence number
    async with _seq_num_lock:
        seq_list_container = _log_seq_num_var.get()
        if seq_list_container is not None:
            current_seq_to_send = seq_list_container[0]
            seq_list_container[0] += 1
        else:
            print(f"ERROR: stream - _log_seq_num_var is None. Cannot get sequence number.")

    try:
        result = await util_client_log(
            seq_num=current_seq_to_send, # Pass the obtained sequence number
            message=message,
            level="INFO",
            logger_name=caller_name,
            request_id=request_id,
            client_id_for_routing=actual_client_id, # Route using actual client_id
            entry_point_name=entry_point_name,
            message_type='stream',
            stream_id=stream_id_param # Pass the provided stream_id separately
        )
        return result
    except Exception as e:
        print(f"ERROR: Failed during async stream call: {e}")
        raise

async def stream_end(stream_id_param: str):
    """Sends a stream_end message to the client, indicating the end of a stream, using a provided stream_id.
    """
    actual_client_id = _client_id_var.get() # Get the actual client ID from context
    request_id = _request_id_var.get()
    entry_point_name = _entry_point_name_var.get() or "unknown_entry_point"

    caller_name = "unknown_caller"
    try:
        frame = inspect.currentframe()
        if frame and frame.f_back:
            caller_name = frame.f_back.f_code.co_name
        del frame
    except Exception as inspect_err:
        print(f"WARNING: Could not inspect caller frame for stream_end: {inspect_err}")

    current_seq_to_send = -1 # Default to an invalid sequence number
    async with _seq_num_lock:
        seq_list_container = _log_seq_num_var.get()
        if seq_list_container is not None:
            current_seq_to_send = seq_list_container[0]
            seq_list_container[0] += 1
        else:
            print(f"ERROR: stream_end - _log_seq_num_var is None. Cannot get sequence number.")

    try:
        result = await util_client_log(
            seq_num=current_seq_to_send, # Pass the obtained sequence number
            message="",
            level="INFO",
            logger_name=caller_name,
            request_id=request_id,
            client_id_for_routing=actual_client_id, # Route using actual client_id
            entry_point_name=entry_point_name,
            message_type='stream_end',
            stream_id=stream_id_param # Pass the provided stream_id separately
        )
        return result
    except Exception as e:
        print(f"ERROR: Failed during async stream_end call: {e}")
        raise


async def client_command(command: str, data: Any = None) -> Any:
    """Sends a command message to the client and waits for a specific acknowledgment and result.

    This function is for commands that require the server to wait for completion
    or a specific response from the client before proceeding.

    Args:
        command: The command string identifier.
        data: Optional JSON-serializable data associated with the command.

    Returns:
        The result returned by the client for the command.

    Raises:
        RuntimeError: If context variables (client_id, request_id) are not set.
        McpError: Propagated from underlying calls if timeouts or client-side errors occur.
    """
    # Get necessary context for routing and correlation
    client_id = _client_id_var.get()
    request_id = _request_id_var.get()
    entry_point_name = _entry_point_name_var.get() # Now needed for logging with seq_num

    if not client_id or not request_id:
        # This should ideally not happen if called within a proper request context
        print(f"ERROR: client_command called without client_id or request_id in context. Command: {command}")
        raise RuntimeError("Client ID or Request ID not found in context for client_command.")

    try:
        # Get current sequence number and increment it for the next call, just like client_log
        current_seq_to_send = -1  # Default to an invalid sequence number
        async with _seq_num_lock:
            seq_list_container = _log_seq_num_var.get()
            if seq_list_container is not None:
                current_seq_to_send = seq_list_container[0]
                seq_list_container[0] += 1
            else:
                print(f"ERROR: client_command - _log_seq_num_var is None. Cannot get sequence number.")

        print(f"INFO: Atlantis: Sending awaitable command '{command}' for client {client_id}, request {request_id}, seq {current_seq_to_send}")
        # Call the dedicated utility function for awaitable commands
        result = await execute_client_command_awaitable(
            client_id_for_routing=client_id,
            request_id=request_id,
            command=command,
            command_data=data,
            seq_num=current_seq_to_send,  # Pass the sequence number
            entry_point_name=entry_point_name  # Pass the entry point name for logging
        )
        print(f"INFO: Atlantis: Received result for awaitable command '{command}': {result}")
        return result
    except Exception as e:
        print(f"ERROR: Atlantis: Error in client_command for command '{command}': {type(e).__name__} - {e}")
        # Re-raise the exception (could be McpError from utils/server or other runtime errors)
        raise

async def client_html(html_content: str, level: str = "INFO"):
    """Sends HTML content back to the requesting client for the current context.
    This is a wrapper around client_log that automatically sets the message type to 'html'.

    HTML messages can be rendered by clients that support HTML display.

    Args:
        html_content: The HTML content to send
        level: Log level (e.g., "INFO", "DEBUG")
    """
    # Send to client_log with message_type set to 'html'
    # client_log is now async and returns a result
    result = await client_log(html_content, level=level, message_type="html")
    return result
