import atlantis
import sqlite3
from datetime import datetime

@app("Mail")
@public
async def message(message:str):
    """
    Send a message to this user
    """

    sender = atlantis.get_user() 

    conn = sqlite3.connect('messages.db')
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            sender TEXT,
            content TEXT,
            created_at TEXT
        )
    """)
    
    current_time_iso = datetime.now().isoformat()
    
    # Attempt to insert the user and their creation time.
    # INSERT OR IGNORE will not insert if the username (PK) already exists.
    cursor.execute(
        "INSERT INTO messages (sender, content, created_at) VALUES (?, ?, ?)",
        (sender, message, current_time_iso)
    )
    
    conn.commit()       
    conn.close()

    await atlantis.client_log(f"Message sent")
