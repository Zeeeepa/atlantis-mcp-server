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

### Function Visibility

**Three layers of visibility control**:

1. **File Mapping (Primary Security Boundary)**
   - Hidden functions are NOT in the mapping
   - Functions not in mapping CANNOT be called
   - Functions not in mapping do NOT appear in tools list

2. **Temporary Overrides (In-Memory, Owner Only)**
   - `_temporarily_visible_functions: Set[str]` - overrides `@hidden`
   - `_temporarily_hidden_functions: Set[str]` - hides visible functions
   - Reset on server restart

3. **Redundant Check (Defense in Depth)**
   - `_get_tools_list()` checks `@hidden` decorator again (line 564)
   - Should never trigger (hidden functions already excluded from mapping)
   - Defensive programming for safety

### Visibility Check Order

In `_create_tools_from_app_mappings()` (line 558-568):

```python
if tool_name in self._temporarily_hidden_functions:
    # 1. Temporary hide (highest priority)
    continue
elif tool_name in self._temporarily_visible_functions:
    # 2. Temporary show (overrides @hidden)
    show_function()
elif decorators_from_info and "hidden" in decorators_from_info:
    # 3. Permanent hide via decorator (should never happen)
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

### Example 2: Calling a Hidden Function (Security)

```
Client: tools/call {name: "hidden_func", args: {...}}
    ↓
server._execute_tool("hidden_func", {...})
    ↓
function_manager.function_call("hidden_func", ...)
    ↓
_find_file_containing_function("hidden_func")
    ↓
_build_function_file_mapping() (if cache stale)
    → File contains @hidden decorator
    → Function NOT added to mapping (line 752 continue)
    ↓
_function_file_mapping.get("hidden_func")
    → Returns: None
    ↓
Raise FileNotFoundError (line 1045)
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

### Test: Hidden Function Cannot Be Called

```python
# Create a hidden function
@hidden
async def test_hidden():
    return "secret"

# Try to call it
result = await function_manager.function_call("test_hidden", ...)
# Expected: FileNotFoundError "Dynamic function 'test_hidden' not found in any file"
```

### Test: Hidden Function Not in Tools List

```python
# Create a hidden function (as above)

# Get tools list
tools = await server._get_tools_list("test")
tool_names = [t.name for t in tools]

# Expected: "test_hidden" NOT in tool_names
assert "test_hidden" not in tool_names
```

### Test: Temporary Visibility Override

```python
# Create a hidden function (as above)

# Show it temporarily (owner only)
await server._execute_tool("_function_show", {"name": "test_hidden"}, ...)

# Check tools list
tools = await server._get_tools_list("test")
tool_names = [t.name for t in tools]

# Expected: "test_hidden" IS in tool_names
assert "test_hidden" in tool_names

# Can now call it
result = await function_manager.function_call("test_hidden", ...)
# Expected: "secret"
```

## FAQ

**Q: Why check for `@hidden` in both the file mapping AND the tools list?**

A: Defense in depth. The file mapping is the primary security boundary, but the tools list has a redundant check (line 564) as defensive programming. If something goes wrong with the mapping, the tools list check provides a backup.

**Q: What happens if I manually add a function to the mapping but it has `@hidden`?**

A: The tools list check would catch it and skip it. However, this scenario shouldn't happen in normal operation since the mapping is built from the same AST parsing that detects `@hidden`.

**Q: Can hidden functions call each other internally?**

A: No, hidden functions cannot be called via the MCP system at all (not in mapping). However, they can be imported and called directly by other Python code within the same module or via normal Python imports.

**Q: What's the difference between `@hidden` and `_temporarily_hidden_functions`?**

A:
- `@hidden`: Permanent, decorator-based, set in code, excluded from file mapping
- `_temporarily_hidden_functions`: Runtime override, in-memory set, owner-controlled, resets on restart

**Q: Why does `_skipped_hidden_functions` exist?**

A: For debugging and introspection. It allows admins to see what functions exist but are hidden. It's not used for any security decisions.
