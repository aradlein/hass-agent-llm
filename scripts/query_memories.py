#!/usr/bin/env python3
"""
Query ChromaDB memories collection for diagnostic purposes.
READ-ONLY investigation script to identify extraneous or problematic memories.
"""

import chromadb
from collections import defaultdict
import json
from datetime import datetime

# Configuration
CHROMADB_HOST = "db.inorganic.me"
CHROMADB_PORT = 8000
COLLECTION_NAME = "home_agent_memories"

# Problematic patterns to detect
TRANSIENT_PATTERNS = [
    "is on", "is off", "are on", "are off",
    "is currently", "are currently", "is now", "are now",
    "temperature is", "humidity is", "status is", "state is",
    "is open", "is closed", "is locked", "is unlocked",
    "is running", "is stopped", "is playing", "is paused"
]

LOW_VALUE_PATTERNS = [
    "conversation occurred", "we discussed", "we talked about",
    "user asked about", "i mentioned", "during the conversation",
    "there is no", "there are no", "no specific",
    "at the time", "the conversation", "in this conversation"
]

NEGATIVE_PATTERNS = [
    "there is no", "there are no", "does not have",
    "no specific", "not available", "not configured"
]


def analyze_memory_content(content, metadata):
    """Analyze a memory for problematic patterns."""
    issues = []
    content_lower = content.lower()

    # Check for transient state patterns
    for pattern in TRANSIENT_PATTERNS:
        if pattern in content_lower:
            issues.append(f"Transient state: '{pattern}'")

    # Check for low-value patterns
    for pattern in LOW_VALUE_PATTERNS:
        if pattern in content_lower:
            issues.append(f"Low-value meta: '{pattern}'")

    # Check for negative patterns
    for pattern in NEGATIVE_PATTERNS:
        if content_lower.startswith(pattern):
            issues.append(f"Negative statement: '{pattern}'")

    # Check word count
    word_count = len([w for w in content.split() if len(w) > 2])
    if word_count < 10:
        issues.append(f"Low word count: {word_count}")

    # Check importance score
    importance = metadata.get('importance', 0)
    if importance < 0.4:
        issues.append(f"Low importance: {importance}")

    return issues


def query_all_memories():
    """Query all memories from ChromaDB and analyze them."""

    print(f"Connecting to ChromaDB at {CHROMADB_HOST}:{CHROMADB_PORT}...")
    print(f"Target collection: {COLLECTION_NAME}\n")

    try:
        client = chromadb.HttpClient(
            host=CHROMADB_HOST,
            port=CHROMADB_PORT,
        )

        collection = client.get_collection(name=COLLECTION_NAME)

        # Get all memories
        result = collection.get(include=["metadatas", "documents"])

        total_memories = len(result['ids'])
        print(f"{'='*80}")
        print(f"TOTAL MEMORIES: {total_memories}")
        print(f"{'='*80}\n")

        if total_memories == 0:
            print("No memories found in collection.")
            return

        # Analysis data structures
        type_counts = defaultdict(int)
        importance_buckets = defaultdict(int)
        problematic_memories = []
        all_memories = []

        # Process each memory
        for i, memory_id in enumerate(result['ids']):
            metadata = result['metadatas'][i] if result['metadatas'] else {}
            content = result['documents'][i] if result['documents'] else ""

            mem_type = metadata.get('type', 'unknown')
            importance = metadata.get('importance', 0)

            type_counts[mem_type] += 1

            # Bucket importance scores
            if importance < 0.3:
                importance_buckets['< 0.3 (very low)'] += 1
            elif importance < 0.5:
                importance_buckets['0.3 - 0.5 (low)'] += 1
            elif importance < 0.7:
                importance_buckets['0.5 - 0.7 (medium)'] += 1
            else:
                importance_buckets['0.7+ (high)'] += 1

            # Analyze for issues
            issues = analyze_memory_content(content, metadata)

            memory_info = {
                'id': memory_id,
                'type': mem_type,
                'importance': importance,
                'content': content,
                'metadata': metadata,
                'issues': issues
            }

            all_memories.append(memory_info)

            if issues:
                problematic_memories.append(memory_info)

        # Print summary statistics
        print("MEMORY TYPE BREAKDOWN:")
        print("-" * 80)
        for mem_type, count in sorted(type_counts.items()):
            percentage = (count / total_memories) * 100
            print(f"  {mem_type:15s}: {count:4d} ({percentage:5.1f}%)")

        print(f"\n{'='*80}")
        print("IMPORTANCE SCORE DISTRIBUTION:")
        print("-" * 80)
        for bucket, count in sorted(importance_buckets.items()):
            percentage = (count / total_memories) * 100
            print(f"  {bucket:20s}: {count:4d} ({percentage:5.1f}%)")

        print(f"\n{'='*80}")
        print(f"PROBLEMATIC MEMORIES: {len(problematic_memories)} / {total_memories} ({(len(problematic_memories)/total_memories)*100:.1f}%)")
        print("=" * 80)

        if problematic_memories:
            print("\nDETAILED PROBLEMATIC MEMORY ANALYSIS:\n")

            # Group by issue type
            issue_groups = defaultdict(list)
            for mem in problematic_memories:
                for issue in mem['issues']:
                    issue_type = issue.split(':')[0]
                    issue_groups[issue_type].append(mem)

            for issue_type, memories in sorted(issue_groups.items()):
                print(f"\n{'-'*80}")
                print(f"ISSUE: {issue_type} ({len(memories)} memories)")
                print(f"{'-'*80}")

                # Show first 5 examples
                for mem in memories[:5]:
                    print(f"\nID: {mem['id']}")
                    print(f"Type: {mem['type']}")
                    print(f"Importance: {mem['importance']:.3f}")
                    print(f"Issues: {', '.join(mem['issues'])}")
                    print(f"Content: {mem['content'][:200]}{'...' if len(mem['content']) > 200 else ''}")

                if len(memories) > 5:
                    print(f"\n... and {len(memories) - 5} more")

        # Show sample of good memories
        good_memories = [m for m in all_memories if not m['issues']]
        if good_memories:
            print(f"\n{'='*80}")
            print(f"SAMPLE GOOD MEMORIES ({len(good_memories)} total):")
            print("=" * 80)

            for mem in good_memories[:5]:
                print(f"\nID: {mem['id']}")
                print(f"Type: {mem['type']}")
                print(f"Importance: {mem['importance']:.3f}")
                print(f"Content: {mem['content'][:200]}{'...' if len(mem['content']) > 200 else ''}")

        # Export detailed JSON report
        report_file = "/workspaces/home-agent/scripts/memory_analysis_report.json"
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_memories": total_memories,
            "type_counts": dict(type_counts),
            "importance_distribution": dict(importance_buckets),
            "problematic_count": len(problematic_memories),
            "problematic_percentage": (len(problematic_memories)/total_memories)*100 if total_memories > 0 else 0,
            "all_memories": all_memories
        }

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n{'='*80}")
        print(f"Detailed JSON report saved to: {report_file}")
        print(f"{'='*80}\n")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    query_all_memories()
