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

@public
async def add(x: float, y: float):
    """Add two numbers. Use for basic addition operations."""
    result = x + y
    await atlantis.client_log(f"{x} + {y} = {result}")
    return result

@hidden
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

- **`@chat`** - Chat functions that get transcript/tools and call LLM
- **`@public`** - Publicly accessible (most common)
- **`@app(name="app_name")`** - Associate with specific app
- **`@location(name="location_name")`** - Associate with location
- **`@shared`** - Persist across reloads
- **`@hidden`** - Hide from tools list

**Combine decorators:**
```python
@app(name="calculator")
@location(name="office")
@public
async def calculate(x: float, y: float):
    """Calculate with app and location context."""
    return x + y
```

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

**Built-in Functions:**
- `_function_show(name)` - Make any function temporarily visible until server restart
- `_function_hide(name)` - Hide any function temporarily until server restart

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

@public
async def create_user(username: str, email: str):
    """Create user account. Use for user registration."""
    return {"user_id": 123, "username": username}

@public
async def get_user(username: str):
    """Get user by username. Use to retrieve user details."""
    return {"username": username, "email": "user@example.com"}

@public
async def delete_user(username: str):
    """Delete user account. Use to remove users."""
    return {"success": True}

@hidden
def _validate_email(email: str):
    """Helper function - not exposed as MCP tool."""
    return "@" in email
```

**Result:** 3 separate MCP tools (`create_user`, `get_user`, `delete_user`) + 1 hidden helper
**Benefits:** Group related functions, share helpers, common imports

### Streaming
```python
@public
async def stream_data():
    """Stream data to client."""
    stream_id = await atlantis.stream_start("data", "stream_data")
    await atlantis.stream("chunk 1", stream_id)
    await atlantis.stream_end(stream_id)
```

### Client Commands
```python
@public
async def get_input():
    """Get input from client."""
    name = await atlantis.client_command("\\input", {"prompt": "Name?"})
    return f"Hello {name}"
```

### State-Dependent Visibility
Use `@hidden` with `_function_show` for state-dependent function visibility, or use `_function_show`/`_function_hide` directly for any function:

```python
@public
async def init_app():
    """Initialize app and show hidden functions."""
    # Initialize your app
    await atlantis.client_log("Initializing app...")

    # Make hidden functions visible
    await atlantis.client_command("_function_show", {"name": "start_service"})
    await atlantis.client_command("_function_show", {"name": "stop_service"})

    return "App initialized"

@hidden
async def start_service():
    """Start service - only visible after initialization."""
    return "Service started"

@hidden
async def stop_service():
    """Stop service - only visible after initialization."""
    return "Service stopped"

# You can also hide any function temporarily
@public
async def hide_debug_functions():
    """Hide debug functions from the tool list."""
    await atlantis.client_command("_function_hide", {"name": "debug_log"})
    await atlantis.client_command("_function_hide", {"name": "test_function"})
    return "Debug functions hidden"
```

**Patterns:**
- Show `init()` first, then reveal other methods after initialization
- Hide debug/test functions when not needed
- Temporarily hide functions during maintenance

### Chat Function
```python
@chat
@public
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