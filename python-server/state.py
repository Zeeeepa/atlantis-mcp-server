#!/usr/bin/env python3
import logging
import os
import sys
from ColoredFormatter import ColoredFormatter

# --- REMOVED basicConfig ---

# Get our app logger
logger = logging.getLogger("mcp_server")
logger.setLevel(logging.DEBUG)

# --- ADDED Handler setup ---
# Create console handler
ch = logging.StreamHandler(sys.stdout) # Use stdout
ch.setLevel(logging.DEBUG) # Process all messages from logger

# Set the custom formatter
ch.setFormatter(ColoredFormatter())

# Add handler to the logger
logger.addHandler(ch)

# Prevent logging from propagating to the root logger
# (important if basicConfig was ever called or might be by libraries)
logger.propagate = False
# --- End Handler setup ---

# Directory to store dynamic function files
FUNCTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dynamic_functions")

# Create functions directory if it doesn't exist
os.makedirs(FUNCTIONS_DIR, exist_ok=True)

# Directory to store dynamic server configs
SERVERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dynamic_servers")
# Create servers directory if it doesn't exist
os.makedirs(SERVERS_DIR, exist_ok=True)

# Server configuration
HOST = "0.0.0.0"  # Listen on all interfaces by default
PORT = 8000

SERVER_REQUEST_TIMEOUT = 30.0 # Seconds to wait for proxied server requests

# Flags to track server state
is_shutting_down = False

