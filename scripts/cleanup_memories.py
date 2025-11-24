#!/usr/bin/env python3
"""
Cleanup problematic memories from ChromaDB.
This script removes low-quality memories identified in the analysis.
"""

import chromadb
from collections import defaultdict

# Configuration
CHROMADB_HOST = "db.inorganic.me"
CHROMADB_PORT = 8000
COLLECTION_NAME = "home_agent_memories"

# IDs of memories to remove based on analysis
MEMORIES_TO_REMOVE = [
    # Transient state: "The kitchen lights are currently on"
    "bcc76961-3cea-4c55-b3e6-5db7cfcd7001",

    # Negative statement: "There is no specific Bed Occupancy Sensor..."
    "87f6b8d1-57b5-46e1-a8f7-eb8ebdecbaa8",

    # Conversation metadata: "The conversation occurred at 20:59..."
    "28ca77c5-d589-47f3-80e4-52cc838c01c1",

    # Event (transient): "The kitchen lights were turned off"
    "ea5d87e2-4600-4286-9565-9baec15f0759",
]

# These have issues but might be worth keeping (user decision)
QUESTIONABLE_MEMORIES = [
    # Low word count but valid facts about birthdays
    ("7c493c51-0cd5-413e-948a-730aadc97520", "anton's birthday is september 28th, 1982"),
    ("e3f1f935-744b-4b8f-9f26-6e3c71f977c6", "Candace's birthday is May 4th, 1989."),
    ("395e5456-4425-45e5-90d6-98afac26f5c4", "Anton's birthday is on September 28th."),

    # Preferences (low word count but potentially valid)
    ("476b251c-dfef-476c-b292-c240ef6349c8", "User prefers kitchen lights at 50% brightness during the day"),
    ("2ed1ce05-0223-4a5a-82ee-6a19bbaedd73", "User prefers kitchen lights to be on at sunrise"),
]


def cleanup_memories(dry_run=True):
    """Remove problematic memories from ChromaDB.

    Args:
        dry_run: If True, only show what would be deleted without actually deleting
    """

    print(f"Connecting to ChromaDB at {CHROMADB_HOST}:{CHROMADB_PORT}...")
    print(f"Target collection: {COLLECTION_NAME}\n")

    try:
        client = chromadb.HttpClient(
            host=CHROMADB_HOST,
            port=CHROMADB_PORT,
        )

        collection = client.get_collection(name=COLLECTION_NAME)

        # Get current count
        result = collection.get()
        total_before = len(result['ids'])
        print(f"Total memories before cleanup: {total_before}\n")

        # Show what will be removed
        print("="*80)
        print(f"{'DRY RUN - ' if dry_run else ''}MEMORIES TO REMOVE: {len(MEMORIES_TO_REMOVE)}")
        print("="*80)

        for memory_id in MEMORIES_TO_REMOVE:
            # Try to get the memory details
            try:
                mem = collection.get(ids=[memory_id], include=["metadatas", "documents"])
                if mem['ids']:
                    content = mem['documents'][0] if mem['documents'] else "N/A"
                    metadata = mem['metadatas'][0] if mem['metadatas'] else {}
                    mem_type = metadata.get('type', 'unknown')
                    importance = metadata.get('importance', 0)

                    print(f"\nID: {memory_id}")
                    print(f"Type: {mem_type}")
                    print(f"Importance: {importance}")
                    print(f"Content: {content}")
                    print(f"Action: {'WOULD DELETE' if dry_run else 'DELETING'}")
                else:
                    print(f"\nID: {memory_id}")
                    print(f"Status: NOT FOUND (may have been already deleted)")
            except Exception as e:
                print(f"\nID: {memory_id}")
                print(f"Error retrieving: {e}")

        # Show questionable memories (not removing automatically)
        print("\n" + "="*80)
        print(f"QUESTIONABLE MEMORIES (review manually): {len(QUESTIONABLE_MEMORIES)}")
        print("="*80)
        print("\nThese memories have issues but might be worth keeping:")
        print("They are NOT being deleted automatically.\n")

        for memory_id, content in QUESTIONABLE_MEMORIES:
            try:
                mem = collection.get(ids=[memory_id], include=["metadatas"])
                if mem['ids']:
                    metadata = mem['metadatas'][0] if mem['metadatas'] else {}
                    mem_type = metadata.get('type', 'unknown')
                    importance = metadata.get('importance', 0)

                    print(f"ID: {memory_id}")
                    print(f"Type: {mem_type}")
                    print(f"Importance: {importance}")
                    print(f"Content: {content}")
                    print(f"Issue: Low word count")
                    print()
            except Exception as e:
                print(f"ID: {memory_id}")
                print(f"Error: {e}\n")

        # Perform deletion if not dry run
        if not dry_run:
            print("\n" + "="*80)
            print("EXECUTING CLEANUP...")
            print("="*80 + "\n")

            deleted_count = 0
            for memory_id in MEMORIES_TO_REMOVE:
                try:
                    collection.delete(ids=[memory_id])
                    print(f"✓ Deleted: {memory_id}")
                    deleted_count += 1
                except Exception as e:
                    print(f"✗ Failed to delete {memory_id}: {e}")

            # Get new count
            result_after = collection.get()
            total_after = len(result_after['ids'])

            print(f"\n{'='*80}")
            print("CLEANUP COMPLETE")
            print("="*80)
            print(f"Memories before: {total_before}")
            print(f"Memories after:  {total_after}")
            print(f"Removed:         {deleted_count}")
            print(f"Expected:        {len(MEMORIES_TO_REMOVE)}")

            if deleted_count == len(MEMORIES_TO_REMOVE):
                print("\n✓ All problematic memories removed successfully!")
            else:
                print(f"\n⚠ Warning: Expected to delete {len(MEMORIES_TO_REMOVE)} but deleted {deleted_count}")

        else:
            print("\n" + "="*80)
            print("DRY RUN COMPLETE - No changes made")
            print("="*80)
            print(f"\nTo actually delete these memories, run:")
            print(f"  python3 {__file__} --execute")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys

    # Check if --execute flag is provided
    execute = "--execute" in sys.argv or "-x" in sys.argv

    if execute:
        print("\n⚠️  WARNING: This will permanently delete memories from the database!")
        response = input("Are you sure you want to proceed? (yes/no): ")
        if response.lower() == "yes":
            cleanup_memories(dry_run=False)
        else:
            print("Cleanup cancelled.")
    else:
        print("Running in DRY RUN mode (no changes will be made)\n")
        cleanup_memories(dry_run=True)
        print(f"\nTo execute the cleanup, run with --execute flag:")
        print(f"  python3 {__file__} --execute")
