#!/usr/bin/env python3
"""Check that a Hugging Face backbone is complete in local storage/cache."""

from __future__ import annotations

import argparse
import sys

from transformers import AutoConfig, AutoTokenizer
from transformers.utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    cached_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    try:
        AutoConfig.from_pretrained(args.model, local_files_only=True)
        AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
        weight_path = None
        for filename in (
            SAFE_WEIGHTS_NAME,
            SAFE_WEIGHTS_INDEX_NAME,
            WEIGHTS_NAME,
            WEIGHTS_INDEX_NAME,
        ):
            try:
                weight_path = cached_file(
                    args.model,
                    filename,
                    local_files_only=True,
                )
            except (OSError, EnvironmentError):
                weight_path = None
            if weight_path is not None:
                break
        if weight_path is None:
            raise FileNotFoundError("no local model weight file or shard index was found")
    except Exception as error:
        if not args.quiet:
            print(f"Local transformer check failed for {args.model!r}: {error}", file=sys.stderr)
        raise SystemExit(2)

    if not args.quiet:
        print(f"Local transformer is complete: {args.model}")


if __name__ == "__main__":
    main()
