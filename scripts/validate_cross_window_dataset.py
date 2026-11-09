#!/usr/bin/env python3
"""Validate a provenance-aware cross-window dataset before training.

The validator is intentionally independent of the model/tokenizer.  It checks
the structural contract that prevents concatenated source documents from
creating artificial relation negatives: every entity has a known source ID,
every gold relation stays within one source, and all offsets/indexes are valid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return payload


def validate_split(path: Path, max_tokens: int | None) -> dict[str, int]:
    documents = _load(path)
    relation_count = 0
    cross_source_relations = 0
    entities_without_source = 0
    for document_id, document in enumerate(documents):
        tokens = document.get("tokens", [])
        entities = document.get("entities", [])
        relations = document.get("relations", [])
        manifest = document.get("source_documents", [])
        if max_tokens is not None and len(tokens) > max_tokens:
            raise ValueError(f"{path.name}[{document_id}] has {len(tokens)} tokens > {max_tokens}")
        if not manifest:
            raise ValueError(f"{path.name}[{document_id}] has no source_documents manifest")
        known_ids = {str(row.get("source_document_id")) for row in manifest}
        for entity_id, entity in enumerate(entities):
            start, end = int(entity["start"]), int(entity["end"])
            if not (0 <= start < end <= len(tokens)):
                raise ValueError(f"{path.name}[{document_id}] entity {entity_id} offset is invalid")
            source_id = entity.get("source_document_id")
            if source_id is None or str(source_id) not in known_ids:
                entities_without_source += 1
        for relation_id, relation in enumerate(relations):
            head, tail = int(relation["head"]), int(relation["tail"])
            if not (0 <= head < len(entities) and 0 <= tail < len(entities)):
                raise ValueError(f"{path.name}[{document_id}] relation {relation_id} index is invalid")
            relation_count += 1
            head_source = entities[head].get("source_document_id")
            tail_source = entities[tail].get("source_document_id")
            if head_source is None or tail_source is None or str(head_source) != str(tail_source):
                cross_source_relations += 1
    if cross_source_relations:
        raise ValueError(f"{path.name} contains {cross_source_relations} cross-source relations")
    return {
        "documents": len(documents),
        "relations": relation_count,
        "entities_without_source": entities_without_source,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()
    summary = {
        split: validate_split(args.dataset_dir / f"{split}.json", args.max_tokens)
        for split in ("train", "dev", "test")
    }
    if any(row["entities_without_source"] for row in summary.values()):
        raise SystemExit("Validation failed: one or more entities have unknown provenance")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
