import atlantis
import sqlite3
from datetime import datetime

@public
async def register():
    """
    Registers a user by adding their username and creation time to 'game.db'.
    Logs if the user is new or was already registered. Errors will bubble up.
    """

    username = atlantis.get_user() 

    await atlantis.client_log(f"Attempting to register user '{username}'...")

    conn = sqlite3.connect('game.db')
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            created_at TEXT
        )
    """)
    
    current_time_iso = datetime.now().isoformat()
    
    # Attempt to insert the user and their creation time.
    # INSERT OR IGNORE will not insert if the username (PK) already exists.
    cursor.execute(
        "INSERT OR IGNORE INTO users (username, created_at) VALUES (?, ?)",
        (username, current_time_iso)
    )
    
    # Check if a row was actually inserted
    newly_registered = cursor.rowcount > 0
    
    conn.commit()       
    conn.close()

    if newly_registered:
        await atlantis.client_log(f"User '{username}' successfully registered at {current_time_iso}.")
    else:
        await atlantis.client_log(f"User '{username}' was already registered.")
        
    # No explicit return, implicitly returns None