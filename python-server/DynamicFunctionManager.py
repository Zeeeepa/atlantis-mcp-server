#!/usr/bin/env python3
"""
Class-based manager for dynamic functions lifecycle and validation.
Implements creating, updating, removing, and validating function code.
"""

import os
import sys
import re
import ast
import json
import asyncio
import logging
import pathlib
import shutil
import datetime
import atlantis
import functools
import inspect
import importlib
import traceback

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from werkzeug.utils import secure_filename

from state import logger

from ColoredFormatter import CYAN, RESET

from mcp.types import (
    Tool,
    TextContent,
    CallToolResult,
    ToolListChangedNotification,
    NotificationParams,
    Annotations,
)

import utils  # Utility module for dynamic functions


# Directory to store dynamic function files
PARENT_PACKAGE_NAME = "dynamic_functions"

# --- Identity Decorator Definition ---
def _mcp_identity_decorator(f):
    """A simple identity decorator that returns the function unchanged. Used as a placeholder for @chat, @public, etc."""
    return f

# --- App Decorator Definition ---
def app(name: str):
    """Decorator to associate a dynamic function with an application name.
    Usage: @app(name="your_app_name")
    """
    def decorator(func_to_decorate):
        setattr(func_to_decorate, '_app_name', name)
        # functools.update_wrapper(decorator, func_to_decorate) # Not strictly needed for AST parsing but good practice
        return func_to_decorate
    return decorator

# --- Location Decorator Definition ---
def location(name: str):
    """Decorator to associate a dynamic function with a location name.
    Usage: @location(name="your_location_name")
    """
    def decorator(func_to_decorate):
        setattr(func_to_decorate, '_location_name', name)
        # functools.update_wrapper(decorator, func_to_decorate) # Not strictly needed for AST parsing but good practice
        return func_to_decorate
    return decorator

# Make the app and location decorators available for dynamic functions to import/use.
# This is a simplified way; a more robust way might involve adding it to a shared module
# that dynamic functions can import from, or injecting it into their global scope upon loading.
# For now, this definition here allows _code_validate_syntax to recognize them by name 'app' and 'location'.

# --- Shared Module Decorator Definition ---
def shared(func_or_module):
    """
    Decorator that marks a function or module as 'shared'.
    When applied, the function/module will not be reloaded when dynamic functions are invalidated.

    Usage: @shared
           def my_persistent_function():
               # This function will maintain its state
               ...
    """
    # Mark the function/module as shared by setting an attribute
    setattr(func_or_module, '_is_shared', True)
    return func_or_module

# --- Hidden Decorator Definition ---
def hidden(func):
    """
    Decorator that marks a function as 'hidden'.
    When applied, the function will not be included in the tools list.

    Usage: @hidden
           def my_hidden_function():
               # This function will not be visible in tools/list
               ...
    """
    # Mark the function as hidden by setting an attribute
    setattr(func, '_is_hidden', True)
    return func

class DynamicFunctionManager:
    def __init__(self, functions_dir):
        # State that was previously global
        self.functions_dir = functions_dir
        self._runtime_errors = {}
        self._dynamic_functions_cache = {}
        self._dynamic_load_lock = asyncio.Lock()

        # NEW: Function-to-file mapping cache
        self._function_file_mapping = {}  # function_name -> filename mapping
        self._function_file_mapping_mtime = 0.0  # track when mapping was last built

        # Create directories if they don't exist
        os.makedirs(self.functions_dir, exist_ok=True)
        self.old_dir = os.path.join(self.functions_dir, "OLD")
        os.makedirs(self.old_dir, exist_ok=True)

    def _validate_app_name(self, app_name: str) -> None:
        """Validate app name format - throw exception for invalid names."""
        if not app_name or not isinstance(app_name, str):
            raise ValueError(f"Invalid app name: {app_name}")
        if re.search(r'[^\w\-]', app_name):
            raise ValueError(f"App name '{app_name}' contains invalid characters. Only letters, numbers, underscores, and dashes are allowed.")

    def _get_app_name_from_path(self, file_path: str) -> str:
        """Extract app name from file path (first subdirectory only).

        Args:
            file_path: Relative path from functions_dir (e.g., 'chat_app/test_chat.py')

        Returns:
            App name from first subdirectory (e.g., 'chat_app')

        Raises:
            ValueError: If file is in root directory or path is invalid
        """
        if not file_path or not isinstance(file_path, str):
            raise ValueError(f"Invalid file path: {file_path}")

        # Split path into components
        path_parts = file_path.split(os.sep)

        # Must have at least 2 parts: subdirectory/filename
        if len(path_parts) < 2:
            raise ValueError(f"File '{file_path}' must be in a subdirectory. Files in root directory are not allowed.")

        # First part is the app name (subdirectory)
        app_name = path_parts[0]

        # Validate the app name
        self._validate_app_name(app_name)

        return app_name

    async def _fs_save_code(self, name: str, code: str, app_name: str = "Home") -> Optional[str]:
        self._validate_app_name(app_name)
        app_dir = os.path.join(self.functions_dir, app_name)
        os.makedirs(app_dir, exist_ok=True)
        file_path = os.path.join(app_dir, f"{name}.py")
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(code)
            logger.debug(f"💾 Saved code for '{name}' to {file_path}")
            return file_path
        except IOError as e:
            logger.error(f"❌ _fs_save_code: Failed to write file {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ _fs_save_code: Unexpected error saving {file_path}: {e}")
            return None

    async def _fs_load_code(self, name: str) -> Optional[str]:
        # Use the mapping to get the relative path
        await self._build_function_file_mapping()
        rel_path = self._function_file_mapping.get(name)
        if not rel_path:
            logger.error(f"❌ _fs_load_code: No mapping found for function '{name}'")
            raise FileNotFoundError(f"Function '{name}' not found in mapping")
        file_path = os.path.join(self.functions_dir, rel_path)
        if not os.path.exists(file_path):
            logger.warning(f"⚠️ _fs_load_code: File not found for '{name}' at {file_path}")
            raise FileNotFoundError(f"Function '{name}' not found at {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            logger.info(f"{CYAN}📋 === LOADING {name} ==={RESET}")
            logger.debug(f"💾 Loaded code for '{name}' from {file_path}")
            return code
        except IOError as e:
            logger.error(f"❌ _fs_load_code: Failed to read file {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ _fs_load_code: Unexpected error loading {file_path}: {e}")
            return None


    # Metadata extraction and validation
    def _code_extract_basic_metadata(self, code_buffer):
        """
        Extracts function name and description using basic regex from a code string buffer.
        Designed to be resilient to minor syntax errors. Returns {'name': ..., 'description': ...}.
        Values can be None if not found.
        """
        metadata = {'name': None, 'description': None}
        if not code_buffer or not isinstance(code_buffer, str):
            return metadata

        # Regex for function name: def optional_async space+ name space* ( ... ):
        func_match = re.search(r'^\s*(async\s+)?def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code_buffer, re.MULTILINE)
        if func_match:
            metadata['name'] = func_match.group(2)
            logger.debug(f"⚙️ Regex extracted name: {metadata['name']}")

            # Regex for the *first* docstring after the function definition line
            # More robust and simpler approach to find docstrings
            # First find the function definition
            fn_def_pattern = r'def\s+' + re.escape(metadata['name']) + r'\s*\(.*?\)\s*:\s*'
            fn_pos = re.search(fn_def_pattern, code_buffer, re.DOTALL)

            docstring_match = None
            if fn_pos:
                # Get the position right after the function signature
                start_pos = fn_pos.end()
                # Look for the first docstring after the function signature
                docstring_pattern = r'\s*"""(.*?)"""'
                docstring_match = re.search(docstring_pattern, code_buffer[start_pos:], re.DOTALL)

                if docstring_match:
                    metadata['description'] = docstring_match.group(1).strip()
                    logger.debug(f"⚙️ Regex extracted description: {metadata['description'][:50]}...")

            # If we couldn't find a docstring after function def, try fallback
            if not docstring_match or not metadata['description']:
                # Fallback: Look for any triple-quoted string near the function definition
                full_func_pattern = f"def\\s+{re.escape(metadata['name'])}.*?\"\"\"(.*?)\"\"\""
                fallback_docstring = re.search(full_func_pattern, code_buffer, re.DOTALL)
                if fallback_docstring:
                    metadata['description'] = fallback_docstring.group(1).strip()
                    logger.debug(f"⚙️ Regex extracted fallback description: {metadata['description'][:50]}...")
                else:
                    # Last resort fallback: any docstring-like pattern in the function
                    simple_fallback = re.search(r'"""(.*?)"""', code_buffer, re.DOTALL)
                    if simple_fallback:
                        metadata['description'] = simple_fallback.group(1).strip()
                        logger.debug(f"⚙️ Regex extracted simple fallback description: {metadata['description'][:50]}...")

        else:
            logger.warning("⚠️ _code_extract_basic_metadata: Could not find function definition via regex.")


        return metadata


    def _ast_node_to_string(self, node: Optional[ast.expr]) -> str:
        """Attempt to reconstruct a string representation of an AST node (for type hints)."""
        if node is None:
            return "Any"
        # Use ast.unparse if available (Python 3.9+) for better accuracy
        if hasattr(ast, 'unparse'):
            try:
                return ast.unparse(node)
            except Exception:
                pass # Fallback to manual reconstruction if unparse fails

        # Manual reconstruction (simplified, fallback for <3.9 or unparse errors)
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return repr(node.value) # e.g., 'None' for NoneType
        elif isinstance(node, ast.Attribute):
            value_str = self._ast_node_to_string(node.value)
            return f"{value_str}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            value_str = self._ast_node_to_string(node.value)
            # Handle slice difference between Python versions
            slice_node = node.slice # Corrected variable name
            if hasattr(ast, 'Index') and isinstance(slice_node, ast.Index): # Python < 3.9
                slice_inner_node = slice_node.value
            else: # Python 3.9+
                slice_inner_node = slice_node

            slice_str = self._ast_node_to_string(slice_inner_node)
            return f"{value_str}[{slice_str}]"
        elif isinstance(node, ast.Tuple): # For Tuple[A, B] or Union[A, B] slices
            elements = ", ".join([self._ast_node_to_string(el) for el in node.elts])
            return f"({elements})" # Representing the structure, not direct type name
        else:
            return "ComplexType"


    def _map_ast_type_to_json_schema(self, annotation_node: Optional[ast.expr]) -> Dict[str, Any]:
        """Maps an AST annotation node to a JSON Schema type component."""
        if annotation_node is None:
            # Default to string if no type hint is provided, as it's common for text-based inputs
            # Alternatively, could use "any" or make it required implicitly if desired.
            return {"type": "string", "description": "Type hint missing, assuming string"}

        # Simple Name types (str, int, etc.)
        if isinstance(annotation_node, ast.Name):
            type_name = annotation_node.id
            if type_name == 'str':
                return {"type": "string"}
            elif type_name == 'int':
                return {"type": "integer"}
            elif type_name == 'float' or type_name == 'complex': # Treat complex as number
                return {"type": "number"}
            elif type_name == 'bool':
                return {"type": "boolean"}
            elif type_name == 'list' or type_name == 'List':
                return {"type": "array"}
            elif type_name == 'dict' or type_name == 'Dict':
                return {"type": "object"}
            elif type_name == 'Any':
                # "any" isn't a standard JSON schema type. Use object without properties? Or skip type field?
                # Let's allow anything but describe it.
                return {"description": "Any type allowed"}
            else:
                # Assume custom object or unhandled simple type
                return {"type": "object", "description": f"Assumed object type: {type_name}"}

        # Constant None (NoneType)
        elif isinstance(annotation_node, ast.Constant) and annotation_node.value is None:
            return {"type": "null"}

        # Subscript types (List[T], Optional[T], Dict[K, V], Union[A, B])
        elif isinstance(annotation_node, ast.Subscript):
            container_node = annotation_node.value
            # Handle slice difference between Python versions
            slice_node = annotation_node.slice # Corrected variable name
            if hasattr(ast, 'Index') and isinstance(slice_node, ast.Index): # Python < 3.9
                slice_inner_node = slice_node.value
            else: # Python 3.9+
                slice_inner_node = slice_node

            container_name = self._ast_node_to_string(container_node) # e.g., 'List', 'Optional', 'Union', 'Dict'

            # Extract inner types from the slice (could be single type or a tuple)
            inner_nodes = []
            if isinstance(slice_inner_node, ast.Tuple):
                inner_nodes = slice_inner_node.elts
            else:
                inner_nodes = [slice_inner_node]

            # Map common container types
            if container_name in ['List', 'list', 'Sequence', 'Iterable', 'Set', 'set']:
                if inner_nodes and inner_nodes[0] is not None:
                    items_schema = self._map_ast_type_to_json_schema(inner_nodes[0])
                    return {"type": "array", "items": items_schema}
                else:
                    return {"type": "array"} # List without specified item type
            elif container_name in ['Dict', 'dict', 'Mapping']:
                if len(inner_nodes) == 2 and inner_nodes[1] is not None:
                    # JSON Schema typically uses additionalProperties for value type
                    value_schema = self._map_ast_type_to_json_schema(inner_nodes[1])
                    # Key type (inner_nodes[0]) is usually string in JSON
                    return {"type": "object", "additionalProperties": value_schema}
                else:
                    return {"type": "object"} # Dict without specified types
            elif container_name == 'Optional':
                if inner_nodes and inner_nodes[0] is not None:
                    schema = self._map_ast_type_to_json_schema(inner_nodes[0])
                    # Make it nullable: allow original type or null
                    existing_types = []
                    if 'type' in schema:
                        existing_types = schema['type'] if isinstance(schema['type'], list) else [schema['type']]
                    elif 'anyOf' in schema: # If inner type was already a Union
                        # Add null to the existing anyOf if not present
                        if not any(t.get('type') == 'null' for t in schema['anyOf']):
                                schema['anyOf'].append({'type': 'null'})
                        return schema
                    else:
                        # Fallback if schema is complex (e.g., just a description)
                        return {'anyOf': [schema, {'type': 'null'}]}

                    if 'null' not in existing_types:
                        existing_types.append('null')
                    schema['type'] = existing_types
                    return schema
                else:
                    # Optional without inner type, allow anything or null
                    return {"type": ["any", "null"], "description":"Optional type specified without inner type"}
            elif container_name == 'Union':
                schemas = [self._map_ast_type_to_json_schema(node) for node in inner_nodes if node is not None]
                # Simplify if it reduces to Optional[T] (Union[T, None])
                non_null_schemas = [s for s in schemas if s.get('type') != 'null']
                has_null = len(schemas) > len(non_null_schemas)

                if len(non_null_schemas) == 1:
                    final_schema = non_null_schemas[0]
                    if has_null: # Make it nullable if None was part of the Union
                        existing_types = []
                        if 'type' in final_schema:
                            existing_types = final_schema['type'] if isinstance(final_schema['type'], list) else [final_schema['type']]
                        elif 'anyOf' in final_schema:
                            if not any(t.get('type') == 'null' for t in final_schema['anyOf']):
                                final_schema['anyOf'].append({'type': 'null'})
                            return final_schema
                        else: # Fallback
                            return {'anyOf': [final_schema, {'type': 'null'}]}

                        if 'null' not in existing_types:
                            existing_types.append('null')
                        final_schema['type'] = existing_types
                    return final_schema
                elif len(non_null_schemas) > 1:
                    # True Union[A, B, ...]
                    result_schema = {"anyOf": non_null_schemas}
                    if has_null: # Add null possibility if None was in the Union
                        result_schema['anyOf'].append({'type': 'null'})
                    return result_schema
                elif has_null: # Only None was in the Union?
                    return {'type': 'null'}
                else: # Empty Union?
                    return {}

            else:
                # Unhandled subscript type (e.g., Tuple[...], custom generics)
                type_str = self._ast_node_to_string(annotation_node)
                return {"description": f"Unhandled generic type: {type_str}"}

        # Fallback for other node types (e.g., ast.BinOp used in type hints?)
        else:
            type_str = self._ast_node_to_string(annotation_node)
            return {"description": f"Unknown type structure: {type_str}"}


    def _ast_arguments_to_json_schema(self, args_node: ast.arguments, docstring: Optional[str] = None) -> Dict[str, Any]:
        """Builds the JSON Schema 'properties' and 'required' fields from AST arguments."""
        properties = {}
        required = []
        parsed_doc_params = {}

        # Basic docstring parsing for parameter descriptions
        if docstring:
            lines = docstring.strip().split('\n')
            param_section = False
            current_param_desc = {}
            for line in lines:
                clean_line = line.strip()
                # Detect start of common param sections
                if clean_line.lower().startswith(('args:', 'arguments:', 'parameters:')):
                    param_section = True
                    continue
                # Stop if we hit return section
                if clean_line.lower().startswith(('returns:', 'yields:')):
                    param_section = False
                    continue

                # Detect common param formats
                # Simple: ":param name: description"
                match_param = re.match(r':param\s+(\w+)\s*:(.*)', clean_line)
                # Typed: "name (type): description" or "name: type\n    description"
                match_typed = re.match(r'(\w+)\s*(?:\(.*\))?\s*:(.*)', clean_line) # Basic check for name: desc

                if match_param:
                    name = match_param.group(1)
                    desc = match_param.group(2).strip()
                    current_param_desc[name] = desc
                    param_section = True # Assume params follow sequentially
                elif param_section and match_typed:
                    name = match_typed.group(1)
                    desc = match_typed.group(2).strip()
                    # If description is empty, it might be on the next line (indented)
                    # This simple parser doesn't handle multi-line descriptions well.
                    current_param_desc[name] = desc if desc else current_param_desc.get(name, '') # Keep previous if empty
                elif param_section and clean_line and not clean_line.startswith(':'):
                    # Assume continuation of the previous param description (basic handling)
                    last_param = next(reversed(current_param_desc), None)
                    if last_param:
                        current_param_desc[last_param] += " " + clean_line

            parsed_doc_params = current_param_desc


        # --- Process Arguments ---
        all_args = args_node.posonlyargs + args_node.args
        num_defaults = len(args_node.defaults)
        defaults_start_index = len(all_args) - num_defaults

        for i, arg in enumerate(all_args):
            name = arg.arg
            param_schema = self._map_ast_type_to_json_schema(arg.annotation)
            param_schema["description"] = parsed_doc_params.get(name, param_schema.get("description", "")) # Add docstring desc

            properties[name] = param_schema

            # Check if it's required (no default value)
            has_default = i >= defaults_start_index
            if not has_default:
                required.append(name)

        # Process kwonlyargs
        for i, arg in enumerate(args_node.kwonlyargs):
            name = arg.arg
            param_schema = self._map_ast_type_to_json_schema(arg.annotation)
            param_schema["description"] = parsed_doc_params.get(name, param_schema.get("description", ""))

            properties[name] = param_schema

            # Check if it's required (kw_defaults[i] is None means no default provided)
            if i < len(args_node.kw_defaults) and args_node.kw_defaults[i] is None:
                required.append(name)
            elif i >= len(args_node.kw_defaults): # Should have a default or None
                required.append(name)

        # Ignore *args (args_node.vararg) and **kwargs (args_node.kwarg)

        return {"properties": properties, "required": required}

    # --- End of Helper Functions ---



    def _code_validate_syntax(self, code_buffer, file_path: Optional[str] = None):
        """
        Validates syntax using ast.parse and extracts info about ALL function definitions found.

        Args:
            code_buffer: The code to validate
            file_path: Optional relative path from functions_dir for app name extraction

        Returns:
            tuple[bool, Optional[str], Optional[List[Dict[str, Any]]]]:
            - is_valid (bool): True if syntax is correct.
            - error_message (Optional[str]): Error details if invalid, None otherwise.
            - functions_info (Optional[List[Dict[str, Any]]]):
                List of dicts with 'name', 'description', 'inputSchema' for each function found,
                None if no functions found or invalid syntax.
        """
        if not code_buffer or not isinstance(code_buffer, str):
            return False, "Empty or invalid code buffer", None

        # Extract app name from file path if provided
        app_name_from_path = None
        if file_path:
            try:
                app_name_from_path = self._get_app_name_from_path(file_path)
                logger.debug(f"📁 Extracted app name from path '{file_path}': {app_name_from_path}")
            except ValueError as e:
                logger.warning(f"⚠️ Could not extract app name from path '{file_path}': {e}")
                # Don't fail validation, just log the warning

        try:
            tree = ast.parse(code_buffer)
            logger.debug("⚙️ Code validation successful (AST parse).")

            functions_info = []
            # Find ALL top-level function definitions
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_def_node = node
                    logger.debug(f"⚙️ Found function definition: {func_def_node.name}")

                    func_name = func_def_node.name
                    docstring = ast.get_docstring(func_def_node)
                    input_schema = {"type": "object"} # Default empty schema

                    # Extract decorators, app_name, and location_name
                    decorator_names = []
                    app_name_from_decorator = None # Initialize app_name
                    location_name_from_decorator = None # Initialize location_name
                    if func_def_node.decorator_list:
                        for decorator_node in func_def_node.decorator_list:
                            if isinstance(decorator_node, ast.Name): # e.g. @public, @hidden
                                decorator_name = decorator_node.id
                                decorator_names.append(decorator_name)
                            elif isinstance(decorator_node, ast.Call): # e.g. @app(name="foo") or @app("foo"), @location(name="bar") or @location("bar")
                                if isinstance(decorator_node.func, ast.Name):
                                    decorator_func_name = decorator_node.func.id
                                    if decorator_func_name == 'app':
                                        # Extract 'name' argument from @app(name="...") or @app("...")
                                        if decorator_node.keywords: # Check keyword arguments like name="foo"
                                            for kw in decorator_node.keywords:
                                                if kw.arg == 'name' and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                                                    if app_name_from_decorator is not None:
                                                        logger.warning(f"⚠️ Multiple @app name specifications for {func_def_node.name}. Using first one: {app_name_from_decorator}")
                                                    else:
                                                        app_name_from_decorator = kw.value.value
                                        # Positional arguments like @app("foo")
                                        if not app_name_from_decorator and decorator_node.args:
                                            if len(decorator_node.args) == 1 and isinstance(decorator_node.args[0], ast.Constant) and isinstance(decorator_node.args[0].value, str):
                                                if app_name_from_decorator is not None: # Should not happen if logic is correct, but for safety
                                                    logger.warning(f"⚠️ Multiple @app name specifications for {func_def_node.name}. Using first one: {app_name_from_decorator}")
                                                else:
                                                    app_name_from_decorator = decorator_node.args[0].value
                                            else:
                                                logger.warning(f"⚠️ @app decorator for {func_def_node.name} has unexpected positional arguments. Expected a single string.")

                                        if app_name_from_decorator is None:
                                            logger.warning(f"⚠️ @app decorator used on {func_def_node.name} but 'name' argument was not found or not a string.")
                                        # We don't add 'app' to decorator_names, as it's handled separately by app_name_from_decorator
                                    elif decorator_func_name == 'location':
                                        # Extract 'name' argument from @location(name="...") or @location("...")
                                        if decorator_node.keywords: # Check keyword arguments like name="foo"
                                            for kw in decorator_node.keywords:
                                                if kw.arg == 'name' and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                                                    if location_name_from_decorator is not None:
                                                        logger.warning(f"⚠️ Multiple @location name specifications for {func_def_node.name}. Using first one: {location_name_from_decorator}")
                                                    else:
                                                        location_name_from_decorator = kw.value.value
                                        # Positional arguments like @location("foo")
                                        if not location_name_from_decorator and decorator_node.args:
                                            if len(decorator_node.args) == 1 and isinstance(decorator_node.args[0], ast.Constant) and isinstance(decorator_node.args[0].value, str):
                                                if location_name_from_decorator is not None: # Should not happen if logic is correct, but for safety
                                                    logger.warning(f"⚠️ Multiple @location name specifications for {func_def_node.name}. Using first one: {location_name_from_decorator}")
                                                else:
                                                    location_name_from_decorator = decorator_node.args[0].value
                                            else:
                                                logger.warning(f"⚠️ @location decorator for {func_def_node.name} has unexpected positional arguments. Expected a single string.")

                                        if location_name_from_decorator is None:
                                            logger.warning(f"⚠️ @location decorator used on {func_def_node.name} but 'name' argument was not found or not a string.")
                                        # We don't add 'location' to decorator_names, as it's handled separately by location_name_from_decorator
                                    else: # It's a call decorator but not 'app' or 'location'
                                        decorator_names.append(decorator_func_name)
                                else: # Decorator call but func is not a simple Name (e.g. @obj.deco())
                                    # Try to reconstruct its name, could be complex e.g. ast.Attribute
                                    # For now, we'll log and skip complex decorator calls for simplicity
                                    logger.debug(f"Skipping complex decorator call structure: {ast.dump(decorator_node.func)}")
                            # Skipping other decorator types for now (e.g. ast.Attribute)

                    # Generate schema from arguments
                    try:
                         schema_parts = self._ast_arguments_to_json_schema(func_def_node.args, docstring)
                         input_schema["properties"] = schema_parts.get("properties", {})
                         input_schema["required"] = schema_parts.get("required", [])
                    except Exception as schema_e:
                         logger.warning(f"⚠️ Could not generate input schema for {func_name}: {schema_e}")
                         input_schema["description"] = f"Schema generation error: {schema_e}"

                    # Resolve app name: decorator takes precedence over folder name
                    final_app_name = app_name_from_decorator
                    if final_app_name is None and app_name_from_path is not None:
                        final_app_name = app_name_from_path
                        logger.debug(f"📁 Using folder-derived app name '{final_app_name}' for function '{func_name}'")
                    elif final_app_name is not None and app_name_from_path is not None and final_app_name != app_name_from_path:
                        logger.warning(f"⚠️ App name conflict for function '{func_name}': decorator='{final_app_name}' vs folder='{app_name_from_path}'. Using decorator value.")

                    function_info = {
                        "name": func_name,
                        "description": docstring or "(No description provided)", # Provide default
                        "inputSchema": input_schema,
                        "decorators": decorator_names, # Add extracted decorators here
                        "app_name": final_app_name, # Add resolved app_name
                        "location_name": location_name_from_decorator # Add extracted location_name
                    }
                    functions_info.append(function_info)

            if functions_info:
                logger.debug(f"⚙️ Found {len(functions_info)} function(s) in file")
                return True, None, functions_info
            else:
                logger.warning("⚠️ Syntax valid, but no top-level function definition found.")
                return True, "Syntax valid, but no function definition found", None

        except SyntaxError as e:
            # Get detailed error information
            error_msg = f"Syntax error at line {e.lineno}, column {e.offset}: {e.msg}"
            if hasattr(e, 'text') and e.text:
                # Show the problematic line if available
                error_msg += f"\nLine content: {e.text.strip()}"
                if e.offset:
                    # Add a pointer to the exact error position
                    error_msg += f"\n{' ' * (e.offset-1)}^"
            logger.warning(f"⚠️ Code validation failed (AST parse): {error_msg}")

            # Try to extract function info using existing regex method as fallback when AST fails
            functions_info = []
            try:
                # Use the existing _code_extract_basic_metadata method
                metadata = self._code_extract_basic_metadata(code_buffer)
                if metadata.get('name'):
                    function_info = {
                        "name": metadata['name'],
                        "description": metadata.get('description') or "(No description provided)",
                        "inputSchema": {"type": "object", "description": "Schema extraction failed due to syntax errors"},
                        "decorators": [],
                        "app_name": app_name_from_path,
                        "location_name": None
                    }
                    functions_info.append(function_info)
                    logger.debug(f"⚙️ Extracted function info for '{metadata['name']}' using existing regex method")
            except Exception as regex_e:
                logger.debug(f"⚠️ Regex fallback also failed: {regex_e}")

            return False, error_msg, functions_info
        except Exception as e:
            error_msg = f"Unexpected error during validation or AST processing: {str(e)}"
            logger.error(f"❌ {error_msg}\n{traceback.format_exc()}") # Log full traceback
            return False, error_msg, None


    def _code_generate_stub(self, name: str) -> str:
        """
        Generates a string containing a basic Python function stub with the given name.
        """
        if not name or not isinstance(name, str):
            name = "unnamed_function" # Default name if invalid

        stub = f"""\
import atlantis

async def {name}():
    \"\"\"
    This is a placeholder function for '{name}'
    \"\"\"
    print(f"Executing placeholder function: {name}...")

    await atlantis.client_log("{name} running")

    # Replace this return statement with your function's result
    return f"Placeholder function '{name}' executed successfully."

"""
        logger.debug(f"⚙️ Generated code stub for function: {name}")
        return stub

    # Cache management
    async def invalidate_function_mapping_cache(self):
        """Invalidate the function-to-file mapping cache."""
        self._function_file_mapping.clear()
        self._function_file_mapping_mtime = 0.0
        logger.debug("🧹 Function-to-file mapping cache invalidated")

    async def _build_function_file_mapping(self):
        """Build the function-to-file mapping by scanning all files."""
        try:
            # Check if we need to rebuild the mapping
            current_mtime = os.path.getmtime(self.functions_dir)
            if (self._function_file_mapping and
                current_mtime == self._function_file_mapping_mtime):
                logger.debug("⚡ Using cached function-to-file mapping")
                return

            logger.info("🔍 Building function-to-file mapping...")
            self._function_file_mapping.clear()

            # Scan all Python files in subdirectories (skip root directory)
            for root, dirs, files in os.walk(self.functions_dir):
                # Skip the root directory itself - only scan subdirectories
                if root == self.functions_dir:
                    continue

                for filename in files:
                    if not filename.endswith('.py'):
                        continue

                    file_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(file_path, self.functions_dir)

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            code = f.read()

                        # Validate and extract function info
                        is_valid, error_message, functions_info = self._code_validate_syntax(code, rel_path)

                        if functions_info:
                            # Add all functions to mapping regardless of validity
                            for func_info in functions_info:
                                func_name = func_info['name']
                                self._function_file_mapping[func_name] = rel_path
                                logger.debug(f"  📍 {func_name} -> {rel_path} (valid: {is_valid})")
                        elif not is_valid:
                            # If syntax is invalid but we can't extract function info,
                            # try to extract function name from filename as fallback
                            filename_without_ext = os.path.splitext(os.path.basename(rel_path))[0]
                            if filename_without_ext and not filename_without_ext.startswith('_'):
                                self._function_file_mapping[filename_without_ext] = rel_path
                                logger.debug(f"  📍 {filename_without_ext} -> {rel_path} (invalid syntax, using filename)")

                    except Exception as e:
                        logger.warning(f"⚠️ Error processing {rel_path} for function mapping: {e}")
                        continue

            self._function_file_mapping_mtime = current_mtime
            logger.info(f"✅ Built function-to-file mapping with {len(self._function_file_mapping)} functions")

            # Periodically clean up orphaned log files (every 10 builds)
            if hasattr(self, '_build_count'):
                self._build_count += 1
            else:
                self._build_count = 1

            if self._build_count % 10 == 0:
                await self._cleanup_orphaned_logs()

        except Exception as e:
            logger.error(f"❌ Error building function-to-file mapping: {e}")

    async def _find_file_containing_function(self, function_name: str) -> Optional[str]:
        """Find which file contains the specified function."""
        await self._build_function_file_mapping()
        return self._function_file_mapping.get(function_name)

    async def invalidate_all_dynamic_module_cache(self):
        """Safely removes ALL dynamic function modules AND the parent package from sys.modules cache."""

        prefix_to_clear = f"{PARENT_PACKAGE_NAME}."
        async with self._dynamic_load_lock:
            # --- Invalidate importlib finder caches ---
            logger.debug("Calling importlib.invalidate_caches()")
            importlib.invalidate_caches()
            # --- End importlib cache invalidation ---

            modules_to_remove = [
                mod for mod in sys.modules
                if mod == PARENT_PACKAGE_NAME or mod.startswith(prefix_to_clear) # Include parent package
            ]
            if modules_to_remove:
                logger.info(f"Invalidating dynamic modules (and parent) from sys.modules cache: {modules_to_remove}")
                for mod_name in modules_to_remove:
                    logger.debug(f"  Attempting to pop: {mod_name}")
                    # Use pop with default None to avoid KeyError if concurrently removed
                    popped_module = sys.modules.pop(mod_name, None)
                    if popped_module:
                        logger.debug(f"    Successfully popped {mod_name}")
                    else:
                        logger.debug(f"    Module {mod_name} not found or already popped.")
                # Log remaining keys after attempted removal
                remaining_dynamic_keys = [k for k in sys.modules if k == PARENT_PACKAGE_NAME or k.startswith(prefix_to_clear)]
                logger.debug(f"  Remaining dynamic keys (incl parent) in sys.modules after pop: {remaining_dynamic_keys}")
            else:
                logger.debug("No dynamic modules (or parent) found in sys.modules to invalidate.")

        # NEW: Also invalidate function mapping cache
        await self.invalidate_function_mapping_cache()

    async def function_add(self, name: str, code: Optional[str] = None, app_name: str = "Home") -> bool:
        self._validate_app_name(app_name)
        file_path = os.path.join(self.functions_dir, app_name, f"{name}.py")
        if os.path.exists(file_path):
            logger.warning(f"Create failed: Function '{name}' already exists in app '{app_name}'.")
            return False
        try:
            code_to_save = code if code is not None else self._code_generate_stub(name)
            if await self._fs_save_code(name, code_to_save, app_name):
                logger.info(f"Function '{name}' created successfully in app '{app_name}'.")
                return True
            else:
                logger.error(f"Create failed: Could not save code for '{name}' in app '{app_name}'.")
                return False
        except Exception as e:
            logger.error(f"Error during function creation for '{name}' in app '{app_name}': {e}")
            logger.debug(traceback.format_exc())
            return False

    def _get_log_file_path(self, rel_path: str) -> str:
        """
        Given a function's relative path (e.g., 'Home/foo.py'), return the corresponding log file path (e.g., 'Home/foo.log').
        """
        if rel_path.endswith('.py'):
            log_rel_path = rel_path[:-3] + '.log'
        else:
            log_rel_path = rel_path + '.log'
        return os.path.join(self.functions_dir, log_rel_path)

    async def _write_error_log(self, name: str, error_message: str, rel_path: str = None) -> None:
        '''
        Write an error message to a function-specific log file in the dynamic_functions folder/subfolder.
        Overwrites any existing log to only keep the latest error.
        Creates a log file named {rel_path}.log with timestamp.
        '''
        if rel_path is None:
            await self._build_function_file_mapping()
            rel_path = self._function_file_mapping.get(name)
            if not rel_path:
                logger.error(f"Cannot write error log: no mapping for function '{name}'")
                return
        log_file_path = self._get_log_file_path(rel_path)
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_content = f"[{timestamp}] ERROR: {error_message}\n"
        try:
            with open(log_file_path, 'w', encoding='utf-8') as f:
                f.write(log_content)
            logger.debug(f"Wrote error log to {log_file_path}")
        except Exception as e:
            logger.error(f"Failed to write error log for '{name}': {e}")

    async def _cleanup_error_log(self, name: str, rel_path: str = None) -> None:
        '''
        Remove the error log file for a function if it exists.
        '''
        if rel_path is None:
            await self._build_function_file_mapping()
            rel_path = self._function_file_mapping.get(name)
            if not rel_path:
                logger.debug(f"Cannot cleanup error log: no mapping for function '{name}'")
                return
        log_file_path = self._get_log_file_path(rel_path)
        if os.path.exists(log_file_path):
            try:
                os.remove(log_file_path)
                logger.debug(f"🧹 Cleaned up error log for '{name}': {log_file_path}")
            except Exception as e:
                logger.warning(f"⚠️ Could not remove error log for '{name}': {e}")

    async def _cleanup_orphaned_logs(self) -> None:
        '''
        Clean up log files for functions that no longer exist.
        '''
        try:
            # Get all .log files in the functions directory and subdirectories
            log_files = []
            for root, dirs, files in os.walk(self.functions_dir):
                for file in files:
                    if file.endswith('.log'):
                        log_files.append(os.path.join(root, file))
            cleaned_count = 0
            for log_file_path in log_files:
                # Get the relative path from functions_dir
                rel_log_path = os.path.relpath(log_file_path, self.functions_dir)
                # Convert .log to .py
                rel_func_path = rel_log_path[:-4] + '.py' if rel_log_path.endswith('.log') else rel_log_path
                func_file_path = os.path.join(self.functions_dir, rel_func_path)
                if not os.path.exists(func_file_path):
                    try:
                        os.remove(log_file_path)
                        cleaned_count += 1
                        logger.debug(f"🧹 Cleaned up orphaned log file: {log_file_path}")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not remove orphaned log file {log_file_path}: {e}")
            if cleaned_count > 0:
                logger.info(f"🧹 Cleaned up {cleaned_count} orphaned log files")
        except Exception as e:
            logger.warning(f"⚠️ Error during orphaned log cleanup: {e}")

    async def function_call(self, name: str, client_id: str, request_id: str, user: str = None, **kwargs) -> Any:
        '''
        Loads and executes a dynamic function by its name, passing kwargs.
        Flushes ALL dynamic function caches before loading to ensure freshness, protected by a lock.
        Ensures parent package exists in sys.modules.
        Returns the function's return value.
        Raises exceptions if the function doesn't exist, fails to load, or errors during execution.
        Gets the 'user' field that tells us who is making the call and passes it to the function context.
        '''
        secure_name = utils.clean_filename(name)
        if not secure_name:
            raise ValueError(f"Invalid function name '{name}' for calling.")

        # NEW: Find which file contains this function
        target_file = await self._find_file_containing_function(name)
        if not target_file:
            raise FileNotFoundError(f"Dynamic function '{name}' not found in any file")

        file_path = os.path.join(self.functions_dir, target_file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dynamic function '{name}' not found at {file_path}")

        context_tokens = None
        # Use the actual filename (without .py) for module name
        module_name = f"{PARENT_PACKAGE_NAME}.{os.path.splitext(target_file)[0]}"
        module = None # Define module outside the lock

        # --- Clear ALL dynamic function child modules from cache FIRST ---
        # This acquires the lock internally, clears, and releases.
        await self.invalidate_all_dynamic_module_cache()
        # --- End Cache Clear ---

        # --- Acquire lock *only* for parent check and specific module loading ---
        async with self._dynamic_load_lock:
            try:
                # --- Ensure Parent Package Exists in sys.modules ---
                if PARENT_PACKAGE_NAME not in sys.modules:
                    logger.info(f"Creating namespace package entry for '{PARENT_PACKAGE_NAME}' in sys.modules")
                    parent_module = importlib.util.module_from_spec(
                        importlib.util.spec_from_loader(PARENT_PACKAGE_NAME, loader=None, is_package=True)
                    )
                    parent_module.__path__ = [self.functions_dir]
                    sys.modules[PARENT_PACKAGE_NAME] = parent_module
                # --- End Parent Package Check ---

                # Clear any previous runtime error for this function before attempting load
                self._runtime_errors.pop(name, None)

                # --- Load the requested module fresh ---
                # Check sys.modules again *inside the lock* in case the watcher re-added it
                # between the invalidate call above and acquiring this lock.
                # Although invalidate_all should have removed it, this is belt-and-suspenders.
                if module_name in sys.modules:
                    logger.warning(f"Module {module_name} unexpectedly found in cache before load, removing again.")
                    del sys.modules[module_name]

                logger.info(f"Loading module fresh: {module_name}")
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Could not create module spec for {target_file}")
                try:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module # Add to sys.modules before exec


                    # Inject identity decorators for known decorator names
                    # This makes @chat, @public, etc., resolvable during module load
                    module.__dict__['chat'] = _mcp_identity_decorator
                    module.__dict__['public'] = _mcp_identity_decorator
                    # Add app decorator which takes parameters
                    module.__dict__['app'] = app
                    # Add location decorator which takes parameters
                    module.__dict__['location'] = location
                    # Add hidden decorator
                    module.__dict__['hidden'] = hidden
                    # Add other known decorator names here if they arise

                    spec.loader.exec_module(module)
                except Exception as load_err:
                    if module_name in sys.modules:
                        del sys.modules[module_name]
                    error_message = f"Error loading module '{module_name}': {load_err}"
                    logger.error(error_message)
                    logger.debug(traceback.format_exc())
                    self._runtime_errors[name] = str(load_err)
                    raise ImportError(error_message) from load_err
                # --- End Load ---

            except Exception as lock_section_err:
                logger.error(f"Unexpected error during locked module handling for {name}: {lock_section_err}")
                logger.debug(traceback.format_exc())
                self._runtime_errors[name] = str(lock_section_err)
                raise

        # --- Lock is released here ---

        # Check if module was loaded successfully inside the lock
        if module is None:
            # This indicates a failure during load that should have raised earlier
            raise RuntimeError(f"Module '{module_name}' failed to load successfully.")

        try:
            # --- Context Setting ---
            bound_client_log = functools.partial(utils.client_log, request_id=request_id, client_id_for_routing=client_id)
            logger.debug(f"Prepared bound_client_log for context. Request ID: {request_id}, Client ID: {client_id}")
            logger.debug(f"Setting context variables via atlantis. User: {user}")

            context_tokens = atlantis.set_context(
                client_log_func=bound_client_log,
                request_id=request_id,
                client_id=client_id,
                user=user,  # Pass the user who made the call - only works if atlantis.py has been updated
                entry_point_name=name # Pass the actual function name (not filename)
            )

            # --- Function Execution ---
            logger.info(f"Attempting to get function '{name}' from loaded module.")
            function_to_call = getattr(module, name, None)
            if not callable(function_to_call):
                raise ValueError(f"No callable function '{name}' found in module '{target_file}'. "
                              f"Please ensure the file contains a function with this name.")

            # Log whether we have user context available
            if user:
                logger.debug(f"Function '{name}' will be called with user context: {user}")

            logger.info(f"Calling dynamic function '{name}' with args: {kwargs.get('args', {})}")

            # Extract args from the kwargs dictionary
            function_args = kwargs.get('args', {})

            if inspect.iscoroutinefunction(function_to_call):
                result = await function_to_call(**function_args)
            else:
                result = function_to_call(**function_args)

            logger.info(f"Dynamic function '{name}' executed successfully.")
            return result

        except Exception as exec_err:
            error_message = f"Error executing dynamic function '{name}': {str(exec_err)}"
            logger.error(error_message)
            logger.debug(traceback.format_exc())
            self._runtime_errors[name] = str(exec_err)
            raise

        finally:
            # --- RESET CONTEXT ---
            if context_tokens:
                logger.debug("Resetting context variables via atlantis")
                atlantis.reset_context(context_tokens)
            else:
                logger.debug("No context tokens found to reset.")


    # --- Function Management Functions --- #

    async def function_validate(self, name: str) -> Dict[str, Any]:
        await self._build_function_file_mapping()
        rel_path = self._function_file_mapping.get(name)
        if not rel_path:
            error_msg = f"No mapping found for function '{name}'"
            await self._write_error_log(name, error_msg, rel_path=None)
            return {'valid': False, 'error': error_msg, 'function_info': None}
        file_path = os.path.join(self.functions_dir, rel_path)
        if not os.path.exists(file_path):
            error_msg = f"Function '{name}' not found at {file_path}"
            await self._write_error_log(name, error_msg, rel_path)
            return {'valid': False, 'error': error_msg, 'function_info': None}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
        except FileNotFoundError as e:
            error_msg = str(e)
            await self._write_error_log(name, error_msg, rel_path)
            return {'valid': False, 'error': error_msg, 'function_info': None}
        is_valid, error_message, functions_info = self._code_validate_syntax(code, rel_path)
        if is_valid:
            logger.info(f"Syntax validation successful for function file '{name}'")
            await self._cleanup_error_log(name, rel_path)
            return {'valid': True, 'error': None, 'function_info': functions_info}
        else:
            error_msg_full = f"Syntax validation failed: {error_message}"
            logger.warning(f"{error_msg_full} Function file: '{name}'")
            await self._write_error_log(name, error_msg_full, rel_path)
            return {'valid': False, 'error': error_message, 'function_info': functions_info}

    async def function_remove(self, name: str) -> bool:
        await self._build_function_file_mapping()
        rel_path = self._function_file_mapping.get(name)
        if not rel_path:
            logger.error(f"Remove failed: No mapping found for function '{name}'")
            return False
        file_path = os.path.join(self.functions_dir, rel_path)
        old_file_path = os.path.join(self.old_dir, f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{os.path.basename(rel_path)}")
        if not os.path.exists(file_path):
            logger.warning(f"⚠️ function_remove: File not found for '{name}' at {file_path}")
            return False
        try:
            os.makedirs(self.old_dir, exist_ok=True)
            shutil.move(file_path, old_file_path)
            logger.info(f"🗑️ Function '{name}' removed. Moved from {file_path} to {old_file_path}")
            await self.invalidate_all_dynamic_module_cache()
            await self._cleanup_error_log(name, rel_path)
            return True
        except Exception as e:
            logger.error(f"❌ function_remove: Failed to remove function '{name}': {e}")
            logger.debug(traceback.format_exc())
            await self._write_error_log(name, f"Failed to remove: {e}", rel_path)
            return False

    async def function_set(self, args: Dict[str, Any], server: Any) -> Tuple[Optional[str], List[TextContent]]:
        """
        Handles the _function_set tool call.
        Extracts all function names using AST parsing, saves the provided code.
        Supports optional filename parameter for multi-function files.
        Returns the filename used (if successful) and a status message.
        Does *not* perform full syntax validation before saving.
        """
        logger.info("⚙️ Handling _function_set call (using AST parsing for all functions)")
        code_buffer = args.get("code")
        target_filename = args.get("filename")  # NEW: Optional filename parameter

        if not code_buffer or not isinstance(code_buffer, str):
            logger.warning("⚠️ function_set: Missing or invalid 'code' parameter.")
            # Return None for name, and the error message
            return None, [TextContent(type="text", text="Error: Missing or invalid 'code' parameter.")]

        # 1. Extract ALL function names using AST parsing
        # Note: For function_set, we don't have a file path yet, so we can't extract folder-based app names
        is_valid, error_message, functions_info = self._code_validate_syntax(code_buffer)

        if not is_valid:
            error_response = f"Error: Could not parse function code: {error_message}"
            logger.warning(f"⚠️ function_set: Failed to parse code via AST.")
            return None, [TextContent(type="text", text=error_response)]

        if not functions_info:
            error_response = "Error: Could not extract any function names from the provided code. Ensure it contains at least one function definition."
            logger.warning(f"⚠️ function_set: No functions found in code.")
            return None, [TextContent(type="text", text=error_response)]

        # Extract function names
        function_names = [func_info['name'] for func_info in functions_info]
        logger.info(f"⚙️ Extracted {len(function_names)} function(s) via AST: {', '.join(function_names)}")

        # 2. Determine filename to save to
        if target_filename:
            # Use specified filename
            filename_to_use = target_filename
            logger.info(f"⚙️ Using specified filename: {filename_to_use}")
        else:
            # Use first function name as filename (backward compatibility)
            filename_to_use = function_names[0]
            logger.info(f"⚙️ Using first function name as filename (backward compatibility): {filename_to_use}")

        # --- Backup existing file before saving new one ---
        secure_name = utils.clean_filename(filename_to_use)
        if secure_name: # Should always be true if filename_to_use is valid
            file_path = os.path.join(self.functions_dir, f"{secure_name}.py")
            if os.path.exists(file_path):
                logger.info(f"💾 Found existing file for '{secure_name}', attempting backup...")
                try:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
                    # Using .py.bak for clarity
                    backup_filename = f"{secure_name}_{timestamp}.py.bak"
                    # Make sure old_dir exists
                    os.makedirs(self.old_dir, exist_ok=True)
                    backup_path = os.path.join(self.old_dir, backup_filename)
                    shutil.copy2(file_path, backup_path) # copy2 preserves metadata
                    logger.info(f"🛡️ Successfully backed up '{secure_name}' to '{backup_path}'")
                except Exception as e:
                    logger.error(f"❌ Failed to backup existing file '{file_path}' to OLD folder: {e}")
                    # Log error but continue, saving the new file might still be desired
            else:
                logger.info(f"ⓘ No existing file found for '{secure_name}', creating new file.")
        else:
            # This case should ideally not happen if name extraction was successful
            logger.warning("⚠️ Could not create secure filename for backup check in function_set.")
        # --- End Backup ---

        # 3. Save the code (validation will happen later when tools are listed/called)
        saved_path = await self._fs_save_code(filename_to_use, code_buffer)

        if not saved_path:
            error_response = f"Error saving functions to file '{filename_to_use}'."
            logger.error(f"❌ function_set: {error_response}")
            # Return filename (as we got this far), but with error message
            return filename_to_use, [TextContent(type="text", text=error_response)]

        logger.info(f"💾 Functions saved successfully to {saved_path}")

        # Clear any cached runtime errors for all functions, as they've been updated
        for func_name in function_names:
            self._runtime_errors.pop(func_name, None)
            # Clean up error logs for functions that were successfully set
            rel_path = self._function_file_mapping.get(func_name)
            if rel_path:
                await self._cleanup_error_log(func_name, rel_path)

        # 4. Attempt AST parsing for immediate feedback (but save regardless)
        syntax_error = None
        try:
            ast.parse(code_buffer)
            logger.info(f"✅ Basic syntax validation (AST parse) successful for '{filename_to_use}'.")
        except SyntaxError as e:
            syntax_error = str(e)
            logger.warning(f"⚠️ Basic syntax validation (AST parse) failed for '{filename_to_use}': {syntax_error}")

        # 5. Clear cache (server needs to reload tools)
        logger.info(f"🧹 Clearing tool cache on server due to function_set for '{filename_to_use}'.")
        server._cached_tools = None
        server._last_functions_dir_mtime = None # Reset mtime to force reload
        server._last_servers_dir_mtime = None # Reset mtime to force reload

        # 6. Prepare success message, including validation status
        save_status = f"Functions saved to '{filename_to_use}.py': {', '.join(function_names)}"
        annotations = None # Default to no annotations
        if syntax_error:
            # If validation failed, add structured error to annotations
            validation_status = f"WARNING: Validation failed."
            response_message = f"{save_status} {validation_status}" # Keep text informative
            annotations = {
                "validationStatus": "ERROR",
                "validationMessage": syntax_error
            }
            logger.warning(f"⚠️ {response_message}")
        else:
            # If validation succeeded
            response_message = f"{save_status} Validation successful."
            logger.info(f"✅ {response_message}")

        # Return TextContent with text and potentially annotations
        return filename_to_use, [TextContent(type="text", text=response_message, annotations=annotations)]

    # Function to get code for a dynamic function
    async def get_function_code(self, args, mcp_server) -> list[TextContent]:
        """
        Get the source code for a dynamic function by name using function-to-file mapping.
        Returns the code as a TextContent object.
        """
        # Get function name
        name = args.get("name")

        # Validate parameters
        if not name:
            raise ValueError("Missing required parameter: name")

        # NEW: Find which file contains this function
        target_file = await self._find_file_containing_function(name)
        if not target_file:
            raise ValueError(f"Function '{name}' not found in any file")

        # Load the code using the existing _fs_load_code utility with the filename
        filename_without_ext = os.path.splitext(target_file)[0]
        code = await self._fs_load_code(filename_without_ext)
        if code is None:
            raise ValueError(f"Function '{name}' not found or could not be read")

        logger.info(f"📋 Retrieved code for function: {name} from {target_file}")

        # Return the code as text content
        return [TextContent(type="text", text=code)]

