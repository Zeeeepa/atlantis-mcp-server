import contextvars
import inspect # Ensure inspect is imported
from typing import Callable, Optional, Any

# --- Context Variables ---
# client_log_func: Holds the partially bound client_log function for the current request
_client_log_var: contextvars.ContextVar[Optional[Callable]] = contextvars.ContextVar("_client_log_var", default=None)
# request_id: The unique ID of the current MCP request
_request_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("_request_id_var", default=None)
# client_id: The ID of the client that initiated the current request
_client_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("_client_id_var", default=None)
# log_seq_num: A counter for log messages within the current request context
_log_seq_num_var: contextvars.ContextVar[int] = contextvars.ContextVar("_log_seq_num_var", default=0)

# entry_point_name: The name of the top-level dynamic function called by the request
_entry_point_name_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("_entry_point_name_var", default=None)
# user: The user who made the call (from the 'user' field in tools/call requests)
_user_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("_user_var", default=None)

# --- Accessor Functions ---

def client_log(message: Any, level: str = "INFO", message_type: str = "text"):
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

        try:
            # Get current sequence number and increment it for the next call
            current_seq = _log_seq_num_var.get()
            _log_seq_num_var.set(current_seq + 1)

            # Call the underlying utils.client_log function, passing seq num, caller name, entry point name, AND message type.
            # This is the key connection between atlantis.client_log and utils.client_log.
            log_func(
                message,
                level=level,
                seq_num=current_seq,
                logger_name=caller_name, # The immediate caller
                entry_point_name=entry_point_name, # The original function called
                message_type=message_type # Type of content (text, json, image, etc.)
            )
        except Exception as e:
            print(f"ERROR: Failed during client_log call (after inspect): {e}")
    else:
        print(f"WARNING: client_log called but no logger in context. Message: {message}")

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
    """Sets the context variables for the current execution scope.
    Returns tokens to be used for resetting.
    """
    client_log_token = _client_log_var.set(client_log_func)
    request_id_token = _request_id_var.set(request_id)
    client_id_token = _client_id_var.set(client_id)
    log_seq_num_token = _log_seq_num_var.set(0)
    # Set the entry point name context
    entry_point_token = _entry_point_name_var.set(entry_point_name)
    # Set the user context
    user_token = _user_var.set(user)
    return (client_log_token, request_id_token, client_id_token, log_seq_num_token, entry_point_token, user_token)

def reset_context(tokens):
    """Resets the context variables using the provided tokens."""
    client_log_token, request_id_token, client_id_token, log_seq_num_token, entry_point_token, user_token = tokens
    _client_log_var.reset(client_log_token)
    _request_id_var.reset(request_id_token)
    _client_id_var.reset(client_id_token)
    _log_seq_num_var.reset(log_seq_num_token)
    # Reset the entry point name context
    _entry_point_name_var.reset(entry_point_token)
    # Reset the user context
    _user_var.reset(user_token)


# --- Utility Functions ---

def client_image(image_path: str, level: str = "INFO", image_format: str = "image/png"):
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
    client_log(prefixed_data, level=level, message_type=image_format)


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
