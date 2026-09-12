from ..db import get_db_connection

def inspect():
    conn = get_db_connection()
    cur = conn.cursor()
    
    print("\n--- POSTS & ASSIGNED EVENTS ---")
    cur.execute("SELECT id, event_id, LEFT(content, 40), created_at FROM posts;")
    for row in cur.fetchall():
        print(f"Post ID: {row[0]} | Event ID: {row[1]} | Content: {row[2]}... | Time: {row[3]}")
        
    print("\n--- EVENT HUBS ---")
    cur.execute("""
        SELECT eh.id, COUNT(p.id) AS post_count, eh.last_updated_at
        FROM event_hubs eh LEFT JOIN posts p ON p.event_id = eh.id
        GROUP BY eh.id, eh.last_updated_at ORDER BY eh.id;
    """)
    for row in cur.fetchall():
        print(f"Event ID: {row[0]} | Post Count: {row[1]} | Updated: {row[2]}")
        
    print("\n--- UNCLUSTERED BUFFER ---")
    cur.execute("SELECT post_id, created_at FROM unclustered_posts_buffer;")
    for row in cur.fetchall():
        print(f"Buffered Post ID: {row[0]} | Time: {row[1]}")
        
    cur.close()
    conn.close()

if __name__ == "__main__":
    inspect()
