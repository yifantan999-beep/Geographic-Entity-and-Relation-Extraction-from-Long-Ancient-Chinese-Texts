"""Document-level, offset-based NER/RE metrics and long-context slices."""

from __future__ import annotations

from typing import Callable, Iterable


def safe_prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "support": tp + fn,
        "predicted": tp + fp,
    }


def entity_key(entity: dict, with_type: bool = True) -> tuple:
    key = (int(entity["start"]), int(entity["end"]))
    return key + (entity["type"],) if with_type else key


def relation_key(relation: dict, entities: list[dict], with_entity_type: bool = True) -> tuple:
    head = entity_key(entities[relation["head"]], with_type=with_entity_type)
    tail = entity_key(entities[relation["tail"]], with_type=with_entity_type)
    if relation.get("symmetric", False) and tail < head:
        head, tail = tail, head
    return head, relation["type"], tail


def _micro_sets(documents: Iterable[dict], selector: Callable[[dict], bool]) -> dict:
    ner_tp = ner_fp = ner_fn = 0
    bound_tp = bound_fp = bound_fn = 0
    re_tp = re_fp = re_fn = 0
    re_bound_tp = re_bound_fp = re_bound_fn = 0
    re_matched_tp = re_matched_fp = re_matched_fn = 0
    exact_ner = exact_re = exact_joint = total = 0
    matched_entity_boundaries = correct_entity_types = 0
    common_relation_endpoint_pairs = exact_relation_type_sets = 0

    for document in documents:
        if not selector(document):
            continue
        total += 1
        gold = document["gold"]
        pred = document["prediction"]
        gold_entities = set(entity_key(e) for e in gold["entities"])
        pred_entities = set(entity_key(e) for e in pred["entities"])
        gold_bounds = set(entity_key(e, False) for e in gold["entities"])
        pred_bounds = set(entity_key(e, False) for e in pred["entities"])
        gold_relations = set(relation_key(r, gold["entities"], True) for r in gold["relations"])
        pred_relations = set(relation_key(r, pred["entities"], True) for r in pred["relations"])
        gold_rel_bounds = set(relation_key(r, gold["entities"], False) for r in gold["relations"])
        pred_rel_bounds = set(relation_key(r, pred["entities"], False) for r in pred["relations"])

        ner_tp += len(gold_entities & pred_entities)
        ner_fp += len(pred_entities - gold_entities)
        ner_fn += len(gold_entities - pred_entities)
        bound_tp += len(gold_bounds & pred_bounds)
        bound_fp += len(pred_bounds - gold_bounds)
        bound_fn += len(gold_bounds - pred_bounds)
        gold_entity_by_bound = {entity_key(e, False): e["type"] for e in gold["entities"]}
        pred_entity_by_bound = {entity_key(e, False): e["type"] for e in pred["entities"]}
        for boundary in gold_bounds & pred_bounds:
            matched_entity_boundaries += 1
            correct_entity_types += int(
                gold_entity_by_bound[boundary] == pred_entity_by_bound[boundary]
            )
        re_tp += len(gold_relations & pred_relations)
        re_fp += len(pred_relations - gold_relations)
        re_fn += len(gold_relations - pred_relations)
        re_bound_tp += len(gold_rel_bounds & pred_rel_bounds)
        re_bound_fp += len(pred_rel_bounds - gold_rel_bounds)
        re_bound_fn += len(gold_rel_bounds - pred_rel_bounds)

        # Relation quality after controlling for endpoint detection: retain only
        # links whose two endpoint boundaries are present in both gold and pred.
        common_bounds = gold_bounds & pred_bounds
        gold_rel_matched = {
            key for key in gold_rel_bounds if key[0] in common_bounds and key[2] in common_bounds
        }
        pred_rel_matched = {
            key for key in pred_rel_bounds if key[0] in common_bounds and key[2] in common_bounds
        }
        re_matched_tp += len(gold_rel_matched & pred_rel_matched)
        re_matched_fp += len(pred_rel_matched - gold_rel_matched)
        re_matched_fn += len(gold_rel_matched - pred_rel_matched)
        gold_types_by_endpoints = {}
        pred_types_by_endpoints = {}
        for key in gold_rel_bounds:
            gold_types_by_endpoints.setdefault((key[0], key[2]), set()).add(key[1])
        for key in pred_rel_bounds:
            pred_types_by_endpoints.setdefault((key[0], key[2]), set()).add(key[1])
        common_endpoints = gold_types_by_endpoints.keys() & pred_types_by_endpoints.keys()
        common_relation_endpoint_pairs += len(common_endpoints)
        exact_relation_type_sets += sum(
            gold_types_by_endpoints[key] == pred_types_by_endpoints[key]
            for key in common_endpoints
        )
        ner_ok = gold_entities == pred_entities
        re_ok = gold_relations == pred_relations
        exact_ner += int(ner_ok)
        exact_re += int(re_ok)
        exact_joint += int(ner_ok and re_ok)

    return {
        "documents": total,
        "ner_strict": safe_prf(ner_tp, ner_fp, ner_fn),
        "ner_boundary": safe_prf(bound_tp, bound_fp, bound_fn),
        "re_strict_end_to_end": safe_prf(re_tp, re_fp, re_fn),
        "re_boundary_end_to_end": safe_prf(re_bound_tp, re_bound_fp, re_bound_fn),
        "re_on_matched_entities": safe_prf(re_matched_tp, re_matched_fp, re_matched_fn),
        "document_exact_match": {
            "ner_accuracy": exact_ner / total if total else 0.0,
            "re_accuracy": exact_re / total if total else 0.0,
            "joint_accuracy": exact_joint / total if total else 0.0,
        },
        "classification_accuracy": {
            "ner_type_on_matched_boundaries": (
                correct_entity_types / matched_entity_boundaries
                if matched_entity_boundaries else 0.0
            ),
            "ner_matched_boundaries": matched_entity_boundaries,
            "re_type_set_on_common_endpoint_pairs": (
                exact_relation_type_sets / common_relation_endpoint_pairs
                if common_relation_endpoint_pairs else 0.0
            ),
            "re_common_endpoint_pairs": common_relation_endpoint_pairs,
        },
    }


def _per_type(documents: list[dict], kind: str) -> dict:
    names: set[str] = set()
    for document in documents:
        names.update(x["type"] for x in document["gold"][kind])
        names.update(x["type"] for x in document["prediction"][kind])
    output = {}
    for name in sorted(names):
        tp = fp = fn = 0
        for document in documents:
            gold = document["gold"]
            pred = document["prediction"]
            if kind == "entities":
                gold_set = {entity_key(e) for e in gold[kind] if e["type"] == name}
                pred_set = {entity_key(e) for e in pred[kind] if e["type"] == name}
            else:
                gold_set = {
                    relation_key(r, gold["entities"], True)
                    for r in gold[kind] if r["type"] == name
                }
                pred_set = {
                    relation_key(r, pred["entities"], True)
                    for r in pred[kind] if r["type"] == name
                }
            tp += len(gold_set & pred_set)
            fp += len(pred_set - gold_set)
            fn += len(gold_set - pred_set)
        output[name] = safe_prf(tp, fp, fn)
    return output


def _distance_metrics(documents: list[dict], bounds: list[int]) -> dict:
    result = {}
    edges = [0, *bounds]
    for lower, upper in zip(edges, [*bounds, None]):
        label = f"{lower}-{upper - 1}" if upper is not None else f">={lower}"
        tp = fp = fn = 0
        for document in documents:
            gold = document["gold"]
            pred = document["prediction"]

            def in_bin(relation: dict, entities: list[dict]) -> bool:
                h = entities[relation["head"]]
                t = entities[relation["tail"]]
                distance = abs((h["start"] + h["end"]) // 2 - (t["start"] + t["end"]) // 2)
                return distance >= lower and (upper is None or distance < upper)

            gs = {relation_key(r, gold["entities"], True) for r in gold["relations"] if in_bin(r, gold["entities"])}
            ps = {relation_key(r, pred["entities"], True) for r in pred["relations"] if in_bin(r, pred["entities"])}
            tp += len(gs & ps)
            fp += len(ps - gs)
            fn += len(gs - ps)
        result[label] = safe_prf(tp, fp, fn)
    return result


def _macro_average(per_type: dict[str, dict]) -> dict:
    supported = [score for score in per_type.values() if score["support"] > 0]
    if not supported:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "types": 0}
    return {
        "precision": sum(score["precision"] for score in supported) / len(supported),
        "recall": sum(score["recall"] for score in supported) / len(supported),
        "f1": sum(score["f1"] for score in supported) / len(supported),
        "types": len(supported),
    }


def _type_confusions(documents: list[dict]) -> dict:
    entity_confusions: dict[str, dict[str, int]] = {}
    relation_confusions: dict[str, dict[str, int]] = {}
    for document in documents:
        gold = document["gold"]
        pred = document["prediction"]
        gold_by_boundary = {entity_key(entity, False): entity["type"] for entity in gold["entities"]}
        pred_by_boundary = {entity_key(entity, False): entity["type"] for entity in pred["entities"]}
        for boundary in gold_by_boundary.keys() & pred_by_boundary.keys():
            gold_type = gold_by_boundary[boundary]
            predicted_type = pred_by_boundary[boundary]
            if gold_type != predicted_type:
                entity_confusions.setdefault(gold_type, {})[predicted_type] = (
                    entity_confusions.setdefault(gold_type, {}).get(predicted_type, 0) + 1
                )

        gold_by_endpoints: dict[tuple, set[str]] = {}
        pred_by_endpoints: dict[tuple, set[str]] = {}
        for relation in gold["relations"]:
            key = relation_key(relation, gold["entities"], False)
            gold_by_endpoints.setdefault((key[0], key[2]), set()).add(key[1])
        for relation in pred["relations"]:
            key = relation_key(relation, pred["entities"], False)
            pred_by_endpoints.setdefault((key[0], key[2]), set()).add(key[1])
        for endpoints in gold_by_endpoints.keys() & pred_by_endpoints.keys():
            gold_types = gold_by_endpoints[endpoints]
            predicted_types = pred_by_endpoints[endpoints]
            if len(gold_types) == 1 and len(predicted_types) == 1:
                gold_type = next(iter(gold_types))
                predicted_type = next(iter(predicted_types))
                if gold_type != predicted_type:
                    relation_confusions.setdefault(gold_type, {})[predicted_type] = (
                        relation_confusions.setdefault(gold_type, {}).get(predicted_type, 0) + 1
                    )
    return {"entities": entity_confusions, "relations": relation_confusions}


def evaluate_documents(
    documents: list[dict],
    long_threshold: int = 256,
    length_buckets: tuple[int, ...] = (128, 256, 512),
    relation_distance_buckets: tuple[int, ...] = (32, 64, 128),
) -> dict:
    """Return overall metrics plus long-document and relation-distance slices."""
    overall = _micro_sets(documents, lambda _: True)
    long_only = _micro_sets(documents, lambda d: d["word_count"] >= long_threshold)
    buckets = {}
    edges = [0, *length_buckets]
    for lower, upper in zip(edges, [*length_buckets, None]):
        label = f"{lower}-{upper - 1}" if upper is not None else f">={lower}"
        buckets[label] = _micro_sets(
            documents,
            lambda d, lo=lower, hi=upper: d["word_count"] >= lo and (hi is None or d["word_count"] < hi),
        )

    observable_gold = recovered_observable = unobservable_gold = recovered_unobservable = 0
    for document in documents:
        pred = document["prediction"]
        pred_keys = {relation_key(r, pred["entities"], True) for r in pred["relations"]}
        gold = document["gold"]
        for relation in gold["relations"]:
            key = relation_key(relation, gold["entities"], True)
            if relation.get("window_observable", True):
                observable_gold += 1
                recovered_observable += int(key in pred_keys)
            else:
                unobservable_gold += 1
                recovered_unobservable += int(key in pred_keys)

    ner_per_type = _per_type(documents, "entities")
    re_per_type = _per_type(documents, "relations")
    return {
        "metric_notes": {
            "ner_strict": "Exact global [start,end) boundary and entity type; micro P/R/F1.",
            "re_strict_end_to_end": "Exact relation type plus both globally offset, typed endpoints.",
            "re_on_matched_entities": "Relation type/endpoints evaluated only where both endpoint boundaries were detected.",
            "document_exact_match": "Fraction of source documents whose complete prediction set equals gold.",
            "classification_accuracy": "Type accuracy after controlling for matched entity boundaries/endpoints.",
            "window_observable": "Both gold relation endpoints co-occur in at least one model window.",
        },
        "overall": overall,
        "long_documents": {"threshold_words": long_threshold, **long_only},
        "length_buckets": buckets,
        "relation_distance_buckets": _distance_metrics(documents, list(relation_distance_buckets)),
        "window_relation_coverage": {
            "observable_gold": observable_gold,
            "observable_recovered": recovered_observable,
            "observable_recall": recovered_observable / observable_gold if observable_gold else 0.0,
            "unobservable_gold": unobservable_gold,
            "unobservable_recovered": recovered_unobservable,
            "unobservable_recall": recovered_unobservable / unobservable_gold if unobservable_gold else 0.0,
            "theoretical_gold_coverage": observable_gold / (observable_gold + unobservable_gold)
            if observable_gold + unobservable_gold else 1.0,
        },
        "ner_per_type": ner_per_type,
        "re_per_type": re_per_type,
        "macro_average": {
            "ner": _macro_average(ner_per_type),
            "re": _macro_average(re_per_type),
        },
        "type_confusions": _type_confusions(documents),
    }
