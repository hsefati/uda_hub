import datetime
import sqlite3
from pathlib import Path
from langchain_core.tools import tool


def _get_db_path(db_name: str) -> str:
    """
    Resolve database paths dynamically regardless of execution context.
    Works from notebook, workflow.py, or direct script execution.
    
    Args:
        db_name: Either 'udahub' or 'cultpass'
    """
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent.parent
    
    if db_name == "udahub":
        db_path = project_root / "uda_hub" / "data" / "core" / "udahub.db"
    elif db_name == "cultpass":
        db_path = project_root / "uda_hub" / "data" / "external" / "cultpass.db"
    else:
        raise ValueError(f"Unknown database: {db_name}")
    
    return str(db_path)


@tool
def search_knowledge_base(query: str) -> str:
    """
    Searches the UDA-Hub knowledge base for policies, FAQs, and troubleshooting steps.
    CRITICAL INSTRUCTION FOR LLM: The query MUST be a single, broad keyword (e.g., "refund", "cancellation", "login").
    Do NOT use full phrases like "refund policy for events" or the database will fail to match.
    """
    try:
        conn = sqlite3.connect(_get_db_path("udahub"))
        cursor = conn.cursor()

        # Using a LIKE search. The LLM is now instructed to pass single words.
        cursor.execute(
            "SELECT title, content FROM knowledge WHERE title LIKE ? OR content LIKE ? OR tags LIKE ?",
            (f"%{query}%", f"%{query}%", f"%{query}%"),
        )
        results = cursor.fetchall()
        conn.close()

        if not results:
            return f"No matching articles found for keyword: '{query}'"

        return "\n\n".join([f"Title: {row[0]}\nContent: {row[1]}" for row in results])
    except Exception as e:
        return f"Database error: {str(e)}"


@tool
def check_reservations(user_id: str) -> str:
    """
    Checks the user's upcoming and past reservations in CultPass.
    Use this when a user asks about an experience, a booking, or a specific event.
    """
    try:
        conn = sqlite3.connect(_get_db_path("cultpass"))
        cursor = conn.cursor()

        # FIXED: Replaced r.reservation_date with r.created_at to match the actual schema
        cursor.execute(
            """
            SELECT e.title, r.status, r.created_at 
            FROM reservations r 
            JOIN experiences e ON r.experience_id = e.experience_id 
            WHERE r.user_id = ?
        """,
            (user_id,),
        )
        results = cursor.fetchall()
        conn.close()

        if not results:
            return "This user has no reservations on file."

        return "\n".join(
            [
                f"Event: {row[0]} | Status: {row[1]} | Booked On: {row[2]}"
                for row in results
            ]
        )
    except Exception as e:
        return f"Database error: {str(e)}"


# Combine them into a list for the Agent
RESEARCHER_TOOLS = [search_knowledge_base, check_reservations]


@tool
def save_ticket_summary(
    ticket_id: str, user_id: str, category: str, tags: list[str], summary: str
):
    """
    Saves the final resolution summary and metadata to the udahub.db.
    Ensures parent records exist in the 'tickets' table first.
    """
    try:
        conn = sqlite3.connect(_get_db_path("udahub"))
        cursor = conn.cursor()

        # 1. Ensure the ticket exists in the parent 'tickets' table
        # We assume a default account_id like 'cultpass' if none is found
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            """
            INSERT OR IGNORE INTO tickets (ticket_id, account_id, user_id, channel, created_at)
            VALUES (?, ?, ?, ?, ?)
        """,
            (ticket_id, "cultpass", user_id, "chat", now),
        )

        # 2. Update ticket_metadata (Using 'main_issue_type' as per schema)
        cursor.execute(
            """
            INSERT INTO ticket_metadata (ticket_id, status, main_issue_type, tags, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticket_id) DO UPDATE SET
                status=excluded.status,
                main_issue_type=excluded.main_issue_type,
                tags=excluded.tags,
                updated_at=excluded.updated_at
        """,
            (
                ticket_id,
                "closed",  # Use 'closed' for durable long-term memory retrieval
                category,
                ", ".join(tags),
                now,
                now,
            ),
        )

        # 3. Log the final summary in ticket_messages
        cursor.execute(
            """
            INSERT INTO ticket_messages (message_id, ticket_id, role, content, created_at)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                f"MSG_SUM_{int(datetime.datetime.now().timestamp())}",
                ticket_id,
                "system",
                f"SUMMARY: {summary}",
                now,
            ),
        )

        conn.commit()
        conn.close()
        return f"Successfully archived ticket {ticket_id}."
    except Exception as e:
        return f"Archiving failed: {str(e)}"


ARCHIVIST_TOOLS = [save_ticket_summary]


@tool
def lookup_user_history(user_id: str):
    """
    Retrieves a summary of the user's past 3 closed support tickets.
    Use this to identify recurring issues or existing preferences based on durable memory.
    """
    conn = sqlite3.connect(_get_db_path("udahub"))
    cursor = conn.cursor()

    # We join tickets, metadata, and messages to get the full story
    # 1. 'tickets' links the user_id to the ticket_id
    # 2. 'ticket_metadata' provides the status and issue type
    # 3. 'ticket_messages' provides the actual summary saved by the Archivist
    query = """
        SELECT t.ticket_id, m.created_at, m.main_issue_type, msg.content 
        FROM tickets t
        JOIN ticket_metadata m ON t.ticket_id = m.ticket_id
        JOIN ticket_messages msg ON t.ticket_id = msg.ticket_id
        WHERE t.user_id = ?
        AND msg.role = 'system'
        AND msg.content LIKE 'SUMMARY:%'
        ORDER BY m.created_at DESC 
        LIMIT 3
    """

    try:
        cursor.execute(query, (user_id,))
        rows = cursor.fetchall()

        if not rows:
            return "No previous ticket history found for this user."

        history_blocks = []
        for row in rows:
            ticket_id, date, issue_type, summary_content = row
            # Strip the 'SUMMARY: ' prefix for a cleaner display
            clean_summary = summary_content.replace("SUMMARY: ", "")
            history_blocks.append(
                f"[[{ticket_id}]][{date}] Category: {issue_type} | Summary: {clean_summary}"
            )

        return "\n---\n".join(history_blocks)
    except Exception as e:
        return f"Error retrieving history: {str(e)}"
    finally:
        conn.close()


HISTORIAN_TOOLS = [lookup_user_history]
