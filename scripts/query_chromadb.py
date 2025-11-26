#!/usr/bin/env python3
"""
Query ChromaDB for a specific entity ID.
READ-ONLY investigation script.
"""

from typing import Any

import chromadb

# Configuration
CHROMADB_HOST = "db.inorganic.me"
CHROMADB_PORT = 8000
COLLECTION_NAME = "home_entities"
ENTITY_TO_FIND = "binary_sensor.motion_sensor_kitchen"


def query_entity_by_id() -> dict[str, Any] | None:
    """Query ChromaDB to check if a specific entity exists."""

    print(f"Connecting to ChromaDB at {CHROMADB_HOST}:{CHROMADB_PORT}...")

    try:
        # Basic connection (adjust if auth/SSL needed)
        client = chromadb.HttpClient(
            host=CHROMADB_HOST,
            port=CHROMADB_PORT,
        )

        print("Connected successfully!")
        print(f"Getting collection: {COLLECTION_NAME}...")

        # Get the collection
        collection = client.get_collection(name=COLLECTION_NAME)

        print(f"Collection found. Querying for entity: {ENTITY_TO_FIND}...")

        # Query for the specific entity ID
        result: dict[str, Any] = collection.get(
            ids=[ENTITY_TO_FIND], include=["metadatas", "documents", "embeddings"]
        )

        # Check if entity exists
        if result and result["ids"]:
            print("\n" + "=" * 60)
            print("✓ ENTITY FOUND!")
            print("=" * 60)
            print(f"\nID: {result['ids'][0]}")

            if result.get("metadatas") and len(result["metadatas"]) > 0:
                metadata = result["metadatas"][0]
                print("\nMetadata:")
                for key, value in metadata.items():
                    print(f"  {key}: {value}")

            if result.get("documents") and len(result["documents"]) > 0:
                print("\nDocument:")
                print(f"  {result['documents'][0]}")

            if result.get("embeddings") and len(result["embeddings"]) > 0:
                embedding = result["embeddings"][0]
                print("\nEmbedding:")
                print(f"  Vector length: {len(embedding)}")
                print(f"  First 5 values: {embedding[:5]}")
                print(f"  Last 5 values: {embedding[-5:]}")
        else:
            print("\n" + "=" * 60)
            print("✗ ENTITY NOT FOUND")
            print("=" * 60)
            print(
                f"\nThe entity '{ENTITY_TO_FIND}' does not exist in collection '{COLLECTION_NAME}'"
            )

        # Additional info: Get collection stats
        print("\n" + "=" * 60)
        print("Collection Statistics:")
        print("=" * 60)
        all_items = collection.get()
        total_count = len(all_items["ids"]) if all_items and "ids" in all_items else 0
        print(f"Total entities in collection: {total_count}")

        # Show sample entity IDs (first 10)
        if all_items and "ids" in all_items and all_items["ids"]:
            print("\nSample entity IDs (first 10):")
            for entity_id in all_items["ids"][:10]:
                print(f"  - {entity_id}")

            # Check for similar sleep number entities
            sleep_entities = [eid for eid in all_items["ids"] if "sleepnumber" in eid.lower()]
            if sleep_entities:
                print(f"\nAll SleepNumber entities in collection ({len(sleep_entities)}):")
                for entity_id in sleep_entities:
                    print(f"  - {entity_id}")

        return result

    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ ERROR")
        print("=" * 60)
        print(f"\n{type(e).__name__}: {e}")
        print("\nTroubleshooting:")
        print("1. Check if ChromaDB server is accessible")
        print(f"2. Verify collection name '{COLLECTION_NAME}' exists")
        print("3. Check if authentication/SSL is required")
        print(f"4. Test with: curl http://{CHROMADB_HOST}:{CHROMADB_PORT}/api/v1/heartbeat")
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("ChromaDB Entity Query Script")
    print("=" * 60)
    print(f"Target: {CHROMADB_HOST}:{CHROMADB_PORT}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Entity: {ENTITY_TO_FIND}")
    print("=" * 60)
    print()

    result = query_entity_by_id()
