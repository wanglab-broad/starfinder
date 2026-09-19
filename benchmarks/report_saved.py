"""Report supplied historical tables without algorithms or implicit experiments.

This selected replacement for repeated comparison/report scripts never treats
legacy timing fields as interchangeable. No original source code is vendored.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def report_saved(quality_path, *, output_dir, keys, variant, timing_path=None,
                 timing_scope, memory_scope):
    """Outer-join unique explicit keys, retaining failures and unmatched rows.

    Callers normalize historical field names explicitly before this boundary.
    No ranking, metric recomputation, missing-value fill, or unit inference.
    """
    if not keys or not variant.strip() or not timing_scope.strip() or not memory_scope.strip():
        raise ValueError('keys, variant and resource scope labels are required')
    quality_path = Path(quality_path)
    table = pd.read_csv(quality_path)
    sources = [quality_path]
    def validate(frame):
        if not set(keys) <= set(frame) or frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError('join keys must be present, non-null and unique')
    validate(table)
    if timing_path is not None:
        timing_path = Path(timing_path)
        timing = pd.read_csv(timing_path)
        validate(timing)
        table = table.merge(timing, on=keys, how='outer', validate='one_to_one',
                            suffixes=('_quality', '_timing'), indicator=True)
        sources.append(timing_path)
    for key, value in [('recipe_variant', variant), ('timing_scope', timing_scope), ('memory_scope', memory_scope)]:
        if key in table:
            raise ValueError(f'reserved report column: {key}')
        table[key] = value
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    table.to_csv(output / 'comparison.csv', index=False)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    (output / 'manifest.json').write_text(json.dumps({'schema_version': 1,
        'sources': [{'path': str(p.resolve()), 'sha256': digest(p)} for p in sources],
        'output_sha256': digest(output / 'comparison.csv'), 'keys': keys, 'variant': variant,
        'timing_scope': timing_scope, 'memory_scope': memory_scope,
        'qualification': 'saved historical values only; no reexecution or scientific validation'}, indent=2)+'\n')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quality', required=True, type=Path)
    parser.add_argument('--timing', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--keys', required=True, nargs='+')
    parser.add_argument('--variant', required=True)
    parser.add_argument('--timing-scope', required=True)
    parser.add_argument('--memory-scope', required=True)
    args = parser.parse_args()
    report_saved(args.quality, output_dir=args.output, keys=args.keys, variant=args.variant,
        timing_path=args.timing, timing_scope=args.timing_scope, memory_scope=args.memory_scope)


if __name__ == '__main__':
    main()
