import atlantis
import sqlite3
from datetime import datetime

@app("Mail")
@public
async def read_messages():
    """
    This is a function for 'read_messages'
    Returns an array of messages sorted by created time so most recent is first
    """
    conn = sqlite3.connect('messages.db')
    cursor = conn.cursor()

    # Create table if it doesn't exist (just to be safe)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            sender TEXT PRIMARY KEY,
            content TEXT,
            created_at TEXT
        )
    """)

    # Query all messages sorted by created_at in descending order
    cursor.execute("SELECT sender, content, created_at FROM messages ORDER BY created_at DESC")
    
    # Fetch all results
    messages = cursor.fetchall()
    
    # Format results as a list of dictionaries for better readability
    formatted_messages = []
    for row in messages:
        formatted_messages.append({
            "sender": row[0],
            "content": row[1],
            "created_at": row[2]
        })
    
    conn.close()
    
    await atlantis.client_log(f"Retrieved {len(formatted_messages)} messages")
    
    return formatted_messages