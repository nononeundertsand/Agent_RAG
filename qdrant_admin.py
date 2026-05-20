import argparse
import os
from typing import Iterable

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FilterSelector


def safe_getattr(obj, name: str, default="-"):
    return getattr(obj, name, default)


def build_client(args: argparse.Namespace) -> QdrantClient:
    url = args.url or os.getenv("QDRANT_HOST") or os.getenv("QDRANT_URL")
    api_key = args.api_key or os.getenv("QDRANT_API_KEY")

    if not url:
        raise ValueError("Missing Qdrant URL. Pass --url or set QDRANT_HOST / QDRANT_URL.")
    if not url.startswith("http"):
        url = "https://" + url

    return QdrantClient(url=url, api_key=api_key, check_compatibility=False)


def selected_collections(client: QdrantClient, names: list[str]) -> list[str]:
    existing = [collection.name for collection in client.get_collections().collections]
    if not names:
        return existing

    missing = sorted(set(names) - set(existing))
    if missing:
        raise ValueError(f"Collections not found: {', '.join(missing)}")
    return names


def print_collection_summary(client: QdrantClient, collection_names: Iterable[str], sample: int):
    for name in collection_names:
        info = client.get_collection(name)
        count = client.count(collection_name=name, exact=True).count
        vectors_count = safe_getattr(info, "vectors_count")
        indexed_vectors_count = safe_getattr(info, "indexed_vectors_count")
        points_count = safe_getattr(info, "points_count", count)
        payload_schema = safe_getattr(info, "payload_schema", {})
        optimizer_status = safe_getattr(info, "optimizer_status")
        print("=" * 88)
        print(f"Collection: {name}")
        print(f"Status: {safe_getattr(info, 'status')}")
        print(f"Points count: {points_count} (exact={count})")
        print(f"Vectors count: {vectors_count}")
        print(f"Indexed vectors count: {indexed_vectors_count}")
        print(f"Optimizer status: {optimizer_status}")
        print(f"Payload schema: {payload_schema}")

        if sample <= 0 or count == 0:
            continue

        points, _ = client.scroll(
            collection_name=name,
            scroll_filter=None,
            limit=sample,
            with_payload=True,
            with_vectors=False,
        )
        print(f"Sample points, limit={sample}:")
        for point in points:
            payload = point.payload or {}
            metadata = payload.get("metadata", payload)
            preview = payload.get("page_content") or payload.get("text") or ""
            preview = str(preview).replace("\n", " ")[:220]
            print(f"- id: {point.id}")
            print(f"  metadata: {metadata}")
            if preview:
                print(f"  preview: {preview}")


def clear_points(client: QdrantClient, collection_names: Iterable[str]):
    for name in collection_names:
        before = client.count(collection_name=name, exact=True).count
        print(f"Clearing all points in collection: {name} ({before} points)")
        client.delete(collection_name=name, points_selector=FilterSelector(filter=Filter(must=[])))
        after = client.count(collection_name=name, exact=True).count
        print(f"Remaining points in {name}: {after}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="View or clear Qdrant collection data. Clear mode deletes points but keeps collections."
    )
    parser.add_argument(
        "--mode",
        choices=["view", "clear"],
        default="view",
        help="view: print collection summaries and sample points; clear: delete all points in selected collections.",
    )
    parser.add_argument("--url", help="Qdrant URL. Can also use QDRANT_HOST or QDRANT_URL env var.")
    parser.add_argument("--api-key", help="Qdrant API key. Can also use QDRANT_API_KEY env var.")
    parser.add_argument(
        "--collection",
        action="append",
        default=[],
        help="Collection to inspect/clear. Repeat this flag for multiple collections. Defaults to all collections.",
    )
    parser.add_argument("--sample", type=int, default=3, help="Number of sample points to print per collection.")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Required in clear mode.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    client = build_client(args)
    collection_names = selected_collections(client, args.collection)

    if not collection_names:
        print("No collections found.")
        return

    print_collection_summary(client, collection_names, sample=args.sample)

    if args.mode == "view":
        return

    if not args.yes:
        raise SystemExit(
            "\nRefusing to clear Qdrant data without --yes. "
            "Re-run with --mode clear --yes if you are sure."
        )

    clear_points(client, collection_names)

    print("\nDone.")


if __name__ == "__main__":
    main()
