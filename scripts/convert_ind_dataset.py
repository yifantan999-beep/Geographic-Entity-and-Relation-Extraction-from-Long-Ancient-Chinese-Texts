"""
Convert the `ind` dataset from original format to CoNLL format expected by ITER.

Original format:
    {"doc_id": "...", "sentence": "raw Chinese text",
     "ner": [["entity_text", "entity_type"], ...],
     "relations": [["head_text", "head_type", "tail_text", "tail_type", "relation_type"], ...]}

Expected format:
    {"tokens": ["char1", "char2", ...],
     "entities": [{"start": 0, "end": 2, "type": "TYPE"}, ...],
     "relations": [{"head": 0, "tail": 1, "type": "REL_TYPE"}, ...]}

Usage:
    cd /path/to/GeoExtraction-main
    python3 scripts/convert_ind_dataset.py
"""

import json
import os
import sys
from collections import defaultdict


def remove_overlapping_entities(entities: list) -> tuple:
    """
    Remove overlapping/nested entities to produce a flat (non-overlapping) set.

    The ITER model with entity_nest_depth=1 requires flat entities where no two
    spans overlap.  We use a greedy longest-first strategy: longer entities are
    preferred, and any entity that overlaps with an already-kept entity is dropped.

    Returns (filtered_entities, keep_mask) where keep_mask[i] is True if the
    original entity at index i was kept.
    """
    if len(entities) <= 1:
        return entities, [True] * len(entities)

    # Sort by length descending, keeping original indices
    indexed = [(e["end"] - e["start"], i, e) for i, e in enumerate(entities)]
    indexed.sort(key=lambda x: x[0], reverse=True)

    keep = set()
    for _, original_idx, entity in indexed:
        # Check if this entity overlaps with any already-kept entity
        overlaps = False
        for kept_idx in keep:
            kept_entity = entities[kept_idx]
            # Two spans overlap if: max(start1, start2) < min(end1, end2)
            if max(entity["start"], kept_entity["start"]) < min(entity["end"], kept_entity["end"]):
                overlaps = True
                break
        if not overlaps:
            keep.add(original_idx)

    # Build filtered list, preserving original order
    keep_mask = [i in keep for i in range(len(entities))]
    filtered = [entities[i] for i in range(len(entities)) if keep_mask[i]]

    return filtered, keep_mask


def convert_items(data: list) -> tuple:
    """Convert a list of data items from old format to new format."""
    converted = []
    total_warnings = 0
    total_dropped_entities = 0
    total_dropped_relations = 0

    for i, item in enumerate(data):
        sentence = item["sentence"]
        find_offsets = defaultdict(int)
        entities = []

        # Convert NER text spans to positional entities
        for ent_text, ent_type in item["ner"]:
            start = sentence.find(ent_text, find_offsets.get(ent_text, 0))
            if start == -1:
                start = sentence.find(ent_text)  # fallback: search from beginning
            if start == -1:
                if total_warnings < 5:
                    print(
                        f"  WARN item {i}: entity '{ent_text}' not found in sentence",
                        file=sys.stderr,
                    )
                total_warnings += 1
                continue
            end = start + len(ent_text)
            find_offsets[ent_text] = end
            entities.append({"start": start, "end": end, "type": ent_type})

        if not entities:
            continue

        # ---- Remove overlapping/nested entities for flat NER ----
        entities, keep_mask = remove_overlapping_entities(entities)
        dropped = sum(1 for k in keep_mask if not k)
        total_dropped_entities += dropped

        # Build old-index -> new-index mapping
        old_to_new = {}
        new_idx = 0
        for old_idx, kept in enumerate(keep_mask):
            if kept:
                old_to_new[old_idx] = new_idx
                new_idx += 1

        # Convert relations from text-based to index-based
        relations = []
        for rel in item.get("relations", []):
            head_text, head_type, tail_text, tail_type, rel_type = rel

            # Find original entity indices
            head_old = None
            tail_old = None
            for j, (et, etype) in enumerate(item["ner"]):
                if et == head_text and etype == head_type and head_old is None:
                    head_old = j
                if et == tail_text and etype == tail_type and tail_old is None:
                    tail_old = j
                if head_old is not None and tail_old is not None:
                    break

            if head_old is None or tail_old is None:
                continue

            # Check if both entities were kept after overlap removal
            if not keep_mask[head_old] or not keep_mask[tail_old]:
                total_dropped_relations += 1
                continue

            relations.append({
                "head": old_to_new[head_old],
                "tail": old_to_new[tail_old],
                "type": rel_type,
            })

        converted.append({
            "tokens": list(sentence),
            "entities": entities,
            "relations": relations,
        })

    return converted, total_warnings, total_dropped_entities, total_dropped_relations


def is_converted(data: list) -> bool:
    """Check if data is already in the target format."""
    return len(data) > 0 and "tokens" in data[0]


def load_json_file(path: str) -> list:
    """Load JSON array from file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(path: str, data: list) -> None:
    """Save JSON array to file, creating parent directories if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    # The code reads from: Path.cwd() / "datasets" / data_dir / filename
    # This is the TARGET we must write to.  Use realpath to follow symlinks.
    target_dir = os.path.realpath("datasets/ind")

    # Raw source is archived separately from the train-ready dataset.
    source_dir = os.path.realpath("datasets/raw/ind_original")
    if not os.path.exists(source_dir):
        source_dir = target_dir

    print(f"Source directory : {source_dir}")
    print(f"Target directory : {target_dir}")
    if source_dir == target_dir:
        print("  (same directory)")
    print()

    # --- Convert each split ---
    for split in ["train", "dev", "test"]:
        target_path = os.path.join(target_dir, f"{split}.json")

        # ---- Step 1: check the TARGET file first ----
        if os.path.exists(target_path):
            target_data = load_json_file(target_path)
            if is_converted(target_data):
                # Check if any items still have overlapping entities
                overlap_count = sum(
                    1 for item in target_data
                    if len(item["entities"]) != len(remove_overlapping_entities(item["entities"])[0])
                )
                if overlap_count == 0:
                    print(f"{split}.json: already converted ({len(target_data)} items), skipping")
                    continue
                else:
                    print(f"{split}.json: has {overlap_count} items with overlaps, re-filtering ...")
                    filtered_data = []
                    total_dropped_ent = 0
                    for item in target_data:
                        ents, keep_mask = remove_overlapping_entities(item["entities"])
                        total_dropped_ent += len(item["entities"]) - len(ents)
                        old_to_new = {}
                        new_idx = 0
                        for old_idx, kept in enumerate(keep_mask):
                            if kept:
                                old_to_new[old_idx] = new_idx
                                new_idx += 1
                        new_rels = []
                        for r in item.get("relations", []):
                            if keep_mask[r["head"]] and keep_mask[r["tail"]]:
                                new_rels.append({
                                    "head": old_to_new[r["head"]],
                                    "tail": old_to_new[r["tail"]],
                                    "type": r["type"],
                                })
                        filtered_data.append({
                            "tokens": item["tokens"],
                            "entities": ents,
                            "relations": new_rels,
                        })
                    save_json_file(target_path, filtered_data)
                    print(f"  Dropped {total_dropped_ent} overlapping entities")
                    print(f"  -> {target_path}  ({len(filtered_data)} items)")
                    print()
                    continue
            # Target exists but has old format -> convert it in-place
            print(f"{split}.json: old format in target ({len(target_data)} items), converting ...")
            converted, warn_count, dropped_ent, dropped_rel = convert_items(target_data)
            save_json_file(target_path, converted)
            if warn_count:
                print(f"  {warn_count} entity-location warning(s)")
            if dropped_ent:
                print(f"  Dropped {dropped_ent} overlapping entities, {dropped_rel} relations")
            print(f"  -> {target_path}  ({len(converted)} items)")
            print()
            continue

        # ---- Step 2: target doesn't exist, look for source data ----
        src_path = os.path.join(source_dir, f"{split}.json")
        if not os.path.exists(src_path):
            print(f"SKIP: neither {target_path} nor {src_path} found")
            continue

        src_data = load_json_file(src_path)

        if is_converted(src_data):
            # Source already in token format — copy to target, but still filter overlaps
            print(f"{split}.json: copying from source ({len(src_data)} items), filtering overlaps ...")
            filtered_data = []
            total_dropped_ent = 0
            for item in src_data:
                ents, keep_mask = remove_overlapping_entities(item["entities"])
                dropped = len(item["entities"]) - len(ents)
                total_dropped_ent += dropped
                # Remap relation indices
                old_to_new = {}
                new_idx = 0
                for old_idx, kept in enumerate(keep_mask):
                    if kept:
                        old_to_new[old_idx] = new_idx
                        new_idx += 1
                new_rels = []
                for r in item.get("relations", []):
                    if keep_mask[r["head"]] and keep_mask[r["tail"]]:
                        new_rels.append({
                            "head": old_to_new[r["head"]],
                            "tail": old_to_new[r["tail"]],
                            "type": r["type"],
                        })
                filtered_data.append({
                    "tokens": item["tokens"],
                    "entities": ents,
                    "relations": new_rels,
                })
            save_json_file(target_path, filtered_data)
            if total_dropped_ent:
                print(f"  Dropped {total_dropped_ent} overlapping entities")
            print(f"  -> {target_path}  ({len(filtered_data)} items)")
        else:
            # Source has old format — convert and write to target
            print(f"{split}.json: converting from source ({len(src_data)} items) ...")
            converted, warn_count, dropped_ent, dropped_rel = convert_items(src_data)
            save_json_file(target_path, converted)
            if warn_count:
                print(f"  {warn_count} entity-location warning(s)")
            if dropped_ent:
                print(f"  Dropped {dropped_ent} overlapping entities, {dropped_rel} relations")
            print(f"  -> {target_path}  ({len(converted)} items)")
        print()

    # --- Copy types.json ---
    types_dst = os.path.join(target_dir, "types.json")
    if not os.path.exists(types_dst):
        types_src = os.path.join(source_dir, "types.json")
        if os.path.exists(types_src):
            import shutil
            os.makedirs(target_dir, exist_ok=True)
            shutil.copy(types_src, types_dst)
            print(f"Copied types.json -> {types_dst}")
    else:
        print(f"types.json: already exists, skipping")
    print()

    # --- Verify ---
    print("=" * 50)
    print("Verification (checking target directory):")
    all_ok = True
    for split in ["train", "dev", "test"]:
        path = os.path.join(target_dir, f"{split}.json")
        if not os.path.exists(path):
            print(f"  {split}.json: MISSING")
            all_ok = False
            continue
        data = load_json_file(path)
        if len(data) == 0:
            print(f"  {split}.json: EMPTY")
            all_ok = False
            continue
        keys = list(data[0].keys())
        ok = "tokens" in keys and "entities" in keys and "relations" in keys
        status = "OK" if ok else "FAILED - still has old format!"
        if not ok:
            all_ok = False
        print(f"  {split}.json: keys={keys}  [{status}]  ({len(data)} items)")

    if all_ok:
        print()
        print("All files converted successfully. Ready to run training!")
    else:
        print()
        print("ERROR: Some files were not converted correctly.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
