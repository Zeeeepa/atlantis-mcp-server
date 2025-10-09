# Dynamic Functions - Technical Architecture

This document explains the internal architecture of the dynamic functions system for developers working on the server codebase.

## Architecture Overview

The dynamic functions system has **one source of truth**: the **file mapping**. This mapping controls both what functions can be called and what functions appear in the tools list.

```
┌─────────────────────────────────────────────────────────────┐
│                    File System (*.py files)                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         _build_function_file_mapping()                       │
│         (DynamicFunctionManager.py:683)                      │
│                                                              │
│  • Scans all .py files recursively                          │
│  • AST parses each file                                     │
│  • Extracts function metadata                               │
│  • EXCLUDES @hidden functions (line 736)                    │
│  • Builds _function_file_mapping dicts                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              File Mapping (SINGLE SOURCE OF TRUTH)           │
│                                                              │
│  _function_file_mapping:         {func_name: file_path}     │
│  _function_file_mapping_by_app:  {app: {func: file_path}}   │
│  _skipped_hidden_functions:      [{name, app, file}, ...]   │
└─────────────┬─────────────────────────────┬─────────────────┘
              │                             │
              ▼                             ▼
┌─────────────────────────┐   ┌─────────────────────────────┐
│   _get_tools_list()     │   │   function_call()           │
│   (server.py:621)       │   │   (DynamicFunctionMgr:239)  │
│                         │   │                             │
│  • Uses file mapping    │   │  • Looks up in file mapping │
│  • Creates Tool objects │   │  • If not found → 404       │
│  • Redundant @hidden    │   │  • If found → load & exec   │
│    check (line 564)     │   │                             │
└─────────────────────────┘   └─────────────────────────────┘
```

## Key Components

### 1. DynamicFunctionManager (`DynamicFunctionManager.py`)

**Purpose**: Manages the lifecycle of dynamic functions

**Key Methods**:

#### `_build_function_file_mapping()` (line 683)
- **Called by**: `_find_file_containing_function()` when cache is stale
- **Frequency**: Only when `dynamic_functions/` directory mtime changes
- **Process**:
  1. Walk through all `.py` files in `dynamic_functions/` recursively
  2. Parse each file with AST (`_code_validate_syntax()`)
  3. Extract function metadata (name, decorators, app_name, location_name)
  4. **Security check** (line 736): If `@hidden` decorator present:
     - Add to `_skipped_hidden_functions` list
     - **Skip adding to mapping** (line 752 `continue`)
  5. Add non-hidden functions to:
     - `_function_file_mapping[func_name] = rel_path`
     - `_function_file_mapping_by_app[app_name][func_name] = rel_path`

#### `_find_file_containing_function(function_name, app_name)` (line 803)
- **Called by**: `_fs_load_code()`, `function_call()`
- **Returns**: File path if function exists in mapping, `None` otherwise
- **Security**: Hidden functions return `None` (not in mapping)

#### `function_call(name, client_id, request_id, user, **kwargs)` (line 968)
- **Called by**: `_execute_tool()` in `server.py`
- **Process**:
  1. Look up function in file mapping via `_find_file_containing_function()`
  2. If not found → raise `FileNotFoundError` (line 1045)
  3. If found → load module, inject decorators, execute function
- **Security**: Hidden functions cannot be called (not in mapping)

### 2. MCPServer (`server.py`)

**Purpose**: MCP protocol handler and tool list provider

**Key Methods**:

#### `_get_tools_list(caller_context)` (line 621)
- **Called by**: MCP protocol `tools/list` handler
- **Frequency**: On demand, with caching based on directory mtime
- **Process**:
  1. Check cache validity (directory mtime)
  2. If invalid, rebuild:
     - Call `_create_tools_from_app_mappings()`
     - Add MCP server tools
     - Add internal tools (`_function_*`, `_server_*`, `_admin_*`)
  3. Return cached tools list

#### `_create_tools_from_app_mappings()` (line 484)
- **Uses**: `self.function_manager._function_file_mapping_by_app` directly
- **Process** (line 498):
  1. Iterate through file mapping (NOT filesystem)
  2. For each function in mapping:
     - Load file and validate (cached per file)
     - Extract all functions from file
     - **Check visibility overrides** (line 558-568):
       - If in `_temporarily_hidden_functions` → skip
       - If in `_temporarily_visible_functions` → show
       - **Redundant check** (line 564): If `@hidden` decorator → skip
         - *Comment: "This should never happen as hidden functions are filtered earlier"*
     - Create Tool object with metadata
     - Add to tools list

#### `_execute_tool(name, args, client_id, ...)` (line 1634)
- **Called by**: MCP protocol `tools/call` handler
- **Process**:
  1. Parse tool name (handle `_function_*`, `_server_*`, MCP proxying)
  2. For regular functions:
     - Call `function_manager.function_call()`
     - Return result to client

## Security Model

### Function Visibility (Opt-In System)

**Default Behavior: Functions are HIDDEN unless decorated with `@visible` or `@public`**

**Two layers of visibility control**:

1. **File Mapping (Primary Security Boundary)** - DynamicFunctionManager.py:646-671
   - Functions WITHOUT `@visible` decorator are NOT in the mapping (unless internal `_function`/`_server`/`_admin`)
   - Functions not in mapping CANNOT be called
   - Functions not in mapping do NOT appear in tools list
   - **Opt-in visibility**: Must use `@visible` or `@public` decorator to expose
   - `@hidden` decorator is **obsolete** (everything is hidden by default)

2. **Redundant Check (Defense in Depth)** - server.py:564
   - `_get_tools_list()` checks decorators again during tool list generation
   - Should never trigger (non-visible functions already excluded from mapping)
   - Defensive programming for safety

**Note:** Temporary visibility overrides (`_function_show`/`_function_hide`) are currently disabled. Use decorators to control visibility.

### Visibility Check Order

In `_build_function_file_mapping()` (DynamicFunctionManager.py:646-671):

```python
# Check decorators during file mapping build
is_internal = func_name.startswith('_function') or func_name.startswith('_server') or func_name.startswith('_admin')
is_visible = "visible" in decorators_from_info if decorators_from_info else False
is_hidden = "hidden" in decorators_from_info if decorators_from_info else False

# Skip if explicitly hidden OR if not visible and not internal
if is_hidden or (not is_visible and not is_internal):
    skip_reason = "hidden by @hidden" if is_hidden else "missing @visible decorator"
    # Function NOT added to mapping - cannot be called!
    continue
```

In `_create_tools_from_app_mappings()` (server.py:558-568):

```python
# Temporary override sets exist but are currently unused (functionality disabled)
if tool_name in self._temporarily_hidden_functions:
    continue
elif tool_name in self._temporarily_visible_functions:
    show_function()
elif decorators_from_info and "hidden" in decorators_from_info:
    # Redundant check (should never happen - already filtered from mapping)
    continue
```

### Owner-Only Tools

Functions starting with `_function`, `_server`, `_admin` are internal:

**Security check** in `_execute_tool()` (line 1702-1718):
```python
if (actual_function_name.startswith('_function') or
    actual_function_name.startswith('_server') or
    actual_function_name.startswith('_admin')):

    caller = user or client_id or "unknown"
    owner = atlantis.get_owner()

    # Localhost websocket connections treated as owner
    if caller.startswith("ws_127.0.0.1_") and owner:
        caller = owner

    if owner and caller != owner:
        raise ValueError("Access denied: Internal functions can only be accessed by owner")
```

## Data Structures

### File Mapping Cache

```python
# In DynamicFunctionManager.__init__():
self._function_file_mapping = {}          # {func_name: rel_path}
self._function_file_mapping_by_app = {}   # {app_name: {func_name: rel_path}}
self._function_file_mapping_mtime = 0.0   # Last build timestamp
self._skipped_hidden_functions = []       # [{name, app, file}, ...]
```

**Invalidation**:
- On directory mtime change
- Via `invalidate_function_mapping_cache()`
- When files are added/removed/modified

### Tools Cache

```python
# In MCPServer.__init__():
self._cached_tools = None                 # List[Tool] or None
self._last_functions_dir_mtime = None     # Last scan timestamp
self._last_servers_dir_mtime = None       # Last scan timestamp
self._last_active_server_keys = set()     # Active MCP servers
```

**Invalidation**:
- On directory mtime change
- When active servers change
- Via `_last_functions_dir_mtime = None`

### Visibility Overrides

```python
# In MCPServer.__init__():
self._temporarily_visible_functions = set()  # Set[str]
self._temporarily_hidden_functions = set()   # Set[str]
```

**Modified by**:
- `_function_show` tool (owner only)
- `_function_hide` tool (owner only)

**Reset by**:
- Server restart

## Call Flow Examples

### Example 1: Calling a Regular Function

```
Client: tools/call {name: "my_function", args: {...}}
    ↓
server._execute_tool("my_function", {...})
    ↓
function_manager.function_call("my_function", ...)
    ↓
_find_file_containing_function("my_function")
    ↓
_build_function_file_mapping() (if cache stale)
    ↓
_function_file_mapping.get("my_function")
    → Returns: "my_function.py"
    ↓
Load module, execute function
    ↓
Return result to client
```

### Example 2: Calling a Non-Visible Function (Security)

```
Client: tools/call {name: "non_visible_func", args: {...}}
    ↓
server._execute_tool("non_visible_func", {...})
    ↓
function_manager.function_call("non_visible_func", ...)
    ↓
_find_file_containing_function("non_visible_func")
    ↓
_build_function_file_mapping() (if cache stale)
    → File does NOT contain @visible decorator
    → Function NOT added to mapping (line 653-671 continue)
    → Logged as: "SKIPPING NON-VISIBLE FUNCTION: non_visible_func (missing @visible decorator)"
    ↓
_function_file_mapping.get("non_visible_func")
    → Returns: None
    ↓
Raise FileNotFoundError (line 964)
    ↓
Return error to client
```

### Example 3: Tools List Generation

```
Client: tools/list {}
    ↓
server._get_tools_list("protocol")
    ↓
Check cache (directory mtime)
    → Cache invalid, rebuild
    ↓
_create_tools_from_app_mappings()
    ↓
Use function_manager._function_file_mapping_by_app
    ↓
For each function in mapping:
    → Load file (cached)
    → Parse AST
    → Check visibility overrides
    → Check @hidden (redundant, defensive)
    → Create Tool object
    ↓
Return tools list to client
```

## Performance Considerations

### Caching Strategy

1. **File Mapping Cache**
   - Built once per directory mtime change
   - Shared across all tool list and function call operations
   - Lightweight (just dictionaries)

2. **Tools List Cache**
   - Built once per directory mtime change
   - Invalidated when active MCP servers change
   - Includes full Tool objects with schemas

3. **File Parsing Cache**
   - Per-file cache within `_create_tools_from_app_mappings()`
   - Prevents redundant AST parsing for multi-function files
   - Cleared on each tools list rebuild

### Bottlenecks

- **AST Parsing**: O(n) where n = number of files
  - Mitigated by caching based on directory mtime
  - Only happens when files actually change

- **Tools List Generation**: O(n) where n = number of functions
  - Mitigated by returning cached list when possible
  - Only regenerates when directory or servers change

## Testing the Security Model

### Test: Non-Visible Function Cannot Be Called

```python
# Create a function without @visible decorator
async def test_non_visible():
    return "secret"

# Try to call it
result = await function_manager.function_call("test_non_visible", ...)
# Expected: FileNotFoundError "Dynamic function 'test_non_visible' not found in any file"
```

### Test: Non-Visible Function Not in Tools List

```python
# Create a function without @visible decorator (as above)

# Get tools list
tools = await server._get_tools_list("test")
tool_names = [t.name for t in tools]

# Expected: "test_non_visible" NOT in tool_names
assert "test_non_visible" not in tool_names
```

### Test: @visible Decorator Makes Function Visible

```python
# Create a function with @visible decorator
@visible
async def test_visible():
    return "public"

# Get tools list
tools = await server._get_tools_list("test")
tool_names = [t.name for t in tools]

# Expected: "test_visible" IS in tool_names
assert "test_visible" in tool_names

# Can call it
result = await function_manager.function_call("test_visible", ...)
# Expected: "public"
```

## FAQ

**Q: Why is visibility opt-in instead of opt-out?**

A: Security by default. Functions must explicitly declare they want to be exposed as MCP tools using `@visible` or `@public`. This prevents accidentally exposing internal helper functions, debug tools, or incomplete code.

**Q: What's the difference between `@visible` and `@public`?**

A:
- `@visible`: Makes function visible in tools list and callable by owner
- `@public`: Makes function visible AND accessible to all users (handled in cloud infrastructure, implies @visible)

**Q: Is `@hidden` still needed?**

A: No, `@hidden` is obsolete. Functions are hidden by default unless decorated with `@visible` or `@public`. You can simply omit the decorator instead of using `@hidden`.

**Q: Can non-visible functions call each other internally?**

A: Yes! Functions without `@visible` cannot be called via MCP, but they can be imported and called directly by other Python code within the same module or via normal Python imports. This makes them perfect for helper functions.

**Q: Why does `_skipped_hidden_functions` exist?**

A: For debugging and introspection. It tracks all functions that were skipped during mapping build (both those with `@hidden` and those missing `@visible`), showing the reason they were skipped. Not used for security decisions.

**Q: Can I change visibility at runtime?**

A: No, the temporary visibility override system (`_function_show`/`_function_hide`) is currently disabled. Use decorators (`@visible` or `@public`) to control function visibility. Changes require editing the function file and rely on the file watcher for automatic reload.
