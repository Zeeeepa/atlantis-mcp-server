#!/usr/bin/env python3
"""
Test script for format_json_log function in utils.py

Tests colorization of various JSON structures including:
- Objects (dicts)
- Arrays (lists)
- Nested structures
- Various data types (strings, numbers, booleans, null)
"""

import sys
import os

# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(__file__))

from utils import format_json_log

def print_test(name, data):
    """Helper to print a test case"""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    print("Input data:")
    print(f"  {repr(data)}")
    print("\nFormatted output (with colors):")
    print(format_json_log(data, colored=True))
    print("\nFormatted output (no colors):")
    print(format_json_log(data, colored=False))


def main():
    print("Testing format_json_log function")
    print("="*60)

    # Test 1: Simple object
    print_test(
        "Simple Object",
        {"name": "Alice", "age": 30, "active": True}
    )

    # Test 2: Simple array of strings
    print_test(
        "Array of Strings",
        ["foo", "bar", "baz"]
    )

    # Test 3: Array of numbers
    print_test(
        "Array of Numbers",
        [1, 2, 3, 42, -5, 3.14]
    )

    # Test 4: Array of mixed types
    print_test(
        "Array of Mixed Types",
        ["hello", 123, True, False, None, -3.14]
    )

    # Test 5: Array of objects
    print_test(
        "Array of Objects",
        [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
            {"name": "Charlie", "age": 35}
        ]
    )

    # Test 6: Object with arrays
    print_test(
        "Object with Arrays",
        {
            "users": ["Alice", "Bob", "Charlie"],
            "scores": [100, 85, 92],
            "flags": [True, False, True],
            "metadata": None
        }
    )

    # Test 7: Nested structure (real-world example similar to MCP tools)
    print_test(
        "Nested Structure (MCP-like)",
        {
            "type": "function",
            "function": {
                "name": "get_user",
                "description": "Fetch user by ID",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {
                            "type": "string",
                            "description": "The user ID"
                        }
                    },
                    "required": ["user_id"]
                }
            }
        }
    )

    # Test 8: Chat transcript-like structure
    print_test(
        "Chat Transcript",
        [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there! How can I help?"}
        ]
    )

    # Test 9: Empty structures
    print_test(
        "Empty Structures",
        {
            "empty_array": [],
            "empty_object": {},
            "null_value": None
        }
    )

    # Test 10: Root-level array with nested objects and arrays
    print_test(
        "Complex Root Array",
        [
            {
                "id": 1,
                "tags": ["python", "json", "colorize"],
                "active": True,
                "score": 95.5
            },
            {
                "id": 2,
                "tags": ["javascript", "api"],
                "active": False,
                "score": None
            }
        ]
    )

    # Test 11: Array of symmetric objects (table-like structure)
    print_test(
        "Table-like Array (Symmetric Objects)",
        [
            {"id": 1, "name": "Alice", "department": "Engineering", "salary": 120000, "active": True},
            {"id": 2, "name": "Bob", "department": "Sales", "salary": 95000, "active": True},
            {"id": 3, "name": "Charlie", "department": "Engineering", "salary": 110000, "active": False},
            {"id": 4, "name": "Diana", "department": "Marketing", "salary": 85000, "active": True},
            {"id": 5, "name": "Eve", "department": "Engineering", "salary": 130000, "active": True}
        ]
    )

    print(f"\n{'='*60}")
    print("All tests completed!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
