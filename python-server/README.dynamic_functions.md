# Dynamic Functions Documentation

## Overview

Create Python functions that become MCP tools automatically. Put `.py` files in `dynamic_functions/` directory.

**Features:**
- Multiple functions per file supported
- Auto-discovery and registration
- Live reloading on file changes
- Each function becomes its own MCP tool

## File Structure

```
dynamic_functions/
├── chat.py              # Single function
├── math_operations.py   # Multiple functions
├── user_management.py   # Related functions grouped
└── OLD/                 # Automatic backups
```

Organize functions however makes sense - one per file or group related functions together.

## Basic Example

```python
import atlantis

@visible
async def add(x: float, y: float):
    """Add two numbers. Use for basic addition operations."""
    result = x + y
    await atlantis.client_log(f"{x} + {y} = {result}")
    return result

# No decorator = hidden by default
async def helper():
    """Helper function - not exposed as tool."""
    return "internal use only"
```

## Requirements

1. Import `atlantis` module
2. Use `async def` (recommended)
3. Add type hints for parameters
4. **CRITICAL**: Docstring becomes AI tool description
5. Use appropriate decorators

## Docstring Guidelines

Write for AI consumption. Be explicit about purpose and when to use.

**Good:**
```python
"""Calculate distance between coordinates using Haversine formula. Use for measuring distances between lat/lng points."""
```

**Bad:**
```python
"""This function does math."""  # Too vague
```

## Decorators

### Visibility (Required)
- **`@visible`** - Make function visible in tools list (REQUIRED for all functions except internal `_function`/`_server`/`_admin`)
- **`@public`** - Make function publicly accessible to all users (handled in cloud, implies @visible)
- **`@protected`** - Make function visible in tools list (acts like @visible, will be expanded with additional features in future)
- **No decorator** - Function is hidden by default, not exposed as tool

### Optional Metadata
- **`@chat`** - Chat functions that get transcript/tools and call LLM
- **`@app(name="app_name")`** - Associate with specific app (DEPRECATED, use folders instead)
- **`@location(name="location_name")`** - Associate with location
- **`@shared`** - Persist across reloads

### Deprecated
- **`@hidden`** - Obsolete (functions are hidden by default without @visible)

**Combine decorators:**
```python
@app(name="calculator")
@location(name="office")
@visible
async def calculate(x: float, y: float):
    """Calculate with app and location context."""
    return x + y
```

## Decorator Behavior

### @visible vs @public vs @protected

Understanding the difference between these decorators is important for access control:

**`@visible`** - Owner-only access:
```python
@visible
async def admin_command(action: str):
    """Execute admin action. Only accessible by function owner."""
    return f"Executing {action}"
```
- Function appears in tools list
- Only the **owner** can call this function
- Use for admin tools, private operations, owner-specific features

**`@public`** - Multi-user access:
```python
@public
async def public_service(query: str):
    """Public API service. Accessible to all users."""
    return f"Result for {query}"
```
- Function appears in tools list (implies `@visible`)
- **Anyone** can call this function (handled in cloud infrastructure)
- Use for shared tools, public APIs, multi-user features
- No need to combine with `@visible` - `@public` includes visibility

**`@protected`** - Protected functions (expandable):
```python
@protected
async def special_function(data: str):
    """Protected function with future access control features."""
    return f"Processing {data}"
```
- Function appears in tools list (currently acts like `@visible`)
- Will be expanded with additional protection features in the future
- Use when you want to mark functions for future access control enhancements

**Access Control Summary:**
- No decorator → Hidden, not callable
- `@visible` → Visible, owner-only
- `@protected` → Visible, owner-only (currently), future enhancements planned
- `@public` → Visible, accessible to all users

## Atlantis Module

**Communication:**
- `client_log(message)` - Send messages to client
- `client_command(command, data)` - Send commands, wait for response
- `client_image(path)`, `client_html(content)`, `client_data(desc, data)` - Send media

**Streaming:**
- `stream_start(sid, who)` → stream_id
- `stream(message, stream_id)` - Send chunks
- `stream_end(stream_id)` - End stream

**Context:**
- `get_caller()`, `get_client_id()`, `get_request_id()`, `get_owner()`

**Utils:**
- `owner_log(message)` - Log to owner file
- `shared` - Persistent memory container (connections, not data)


## Shared Container

Use `atlantis.shared` for persistent memory objects (connections, not data).

```python
# Initialize database connection once
if not atlantis.shared.get("db"):
    atlantis.shared.set("db", sqlite3.connect("app.db"))

db = atlantis.shared.get("db")
```

**Store:** DB connections, API clients, caches
**Don't store:** User data, application data (use databases)

**Methods:** `shared.get(key)`, `shared.set(key, value)`, `shared.remove(key)`, `shared.keys()`

## Examples

### Multiple Functions Per File
You can put many functions in one file - each becomes its own MCP tool:

```python
# File: user_management.py
import atlantis

@visible
async def create_user(username: str, email: str):
    """Create user account. Use for user registration."""
    return {"user_id": 123, "username": username}

@visible
async def get_user(username: str):
    """Get user by username. Use to retrieve user details."""
    return {"username": username, "email": "user@example.com"}

@visible
async def delete_user(username: str):
    """Delete user account. Use to remove users."""
    return {"success": True}

# No decorator = hidden by default
def _validate_email(email: str):
    """Helper function - not exposed as MCP tool."""
    return "@" in email
```

**Result:** 3 separate MCP tools (`create_user`, `get_user`, `delete_user`) + 1 hidden helper
**Benefits:** Group related functions, share helpers, common imports

### Streaming
```python
@visible
async def stream_data():
    """Stream data to client."""
    stream_id = await atlantis.stream_start("data", "stream_data")
    await atlantis.stream("chunk 1", stream_id)
    await atlantis.stream_end(stream_id)
```

### Client Commands
```python
@visible
async def get_input():
    """Get input from client."""
    name = await atlantis.client_command("\\input", {"prompt": "Name?"})
    return f"Hello {name}"
```

### Helper Functions
Functions without `@visible` are hidden by default - perfect for internal helpers:

```python
@visible
async def process_data(data: str):
    """Process and validate data."""
    # Use internal helper functions
    if not _validate_data(data):
        return "Invalid data"

    cleaned = _clean_data(data)
    return f"Processed: {cleaned}"

# No decorator = hidden by default, not exposed as MCP tool
def _validate_data(data: str):
    """Internal helper - validates data format."""
    return len(data) > 0 and data.strip() != ""

# No decorator = hidden by default, not exposed as MCP tool
def _clean_data(data: str):
    """Internal helper - cleans and formats data."""
    return data.strip().lower()
```

**Patterns:**
- Keep helper/utility functions without decorators (hidden by default)
- Only add `@visible` to functions that should be MCP tools
- Internal functions can still be called by visible functions

### Chat Function
```python
@chat
@visible
async def chat():
    """Chat function that processes conversation and calls LLM."""
    # Get conversation history
    transcript = await atlantis.client_command("\\transcript get")

    # Get available tools
    tools = await atlantis.client_command("\\transcript tools")

    # Call your LLM with transcript and tools
    response = await call_llm(transcript, tools)

    # Stream response back
    stream_id = await atlantis.stream_start("chat", "ai_assistant")
    await atlantis.stream(response, stream_id)
    await atlantis.stream_end(stream_id)
```

## Type Hints

Type hints generate JSON schemas automatically:
```python
def func(text: str, items: List[str], optional: Optional[int] = None):
    """Function with type hints."""
    pass
```

## Best Practices

- Use `async def` for functions
- Add type hints for parameters
- Write clear docstrings for AI
- Use `atlantis.shared` for connections only
- Group related functions in same file
- Use descriptive names

## Troubleshooting

**Function not showing:** Check syntax, decorators, file location
**Execution errors:** Check `.log` files in `dynamic_functions/`
**Context issues:** Use `await` with atlantis methods

Functions automatically become MCP tools when saved to `dynamic_functions/`.