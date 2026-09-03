#!/usr/bin/env python3
"""Run radar_gp_state_estimation.py once per sequence, reusing one config.yaml.

Edit DATA_ROOT / SEQUENCES below, then: python run_batch.py
"""
import argparse
import os
import os.path as osp
import re
import subprocess
import sys
import time

# Sequences to run. Names are joined with DATA_ROOT; absolute paths are used as-is.
DATA_ROOT = '/home/dl/Documents/phd/data/boreas'
SEQUENCES = [
    'warthog-dome-1',
    'warthog-dome-2',
    'warthog-dome-3',
    'warthog-dome-4',
    'warthog-dome-5',
    'warthog-woody-1',
    'warthog-woody-2',
    'warthog-woody-3',
    'warthog-woody-4',
    'warthog-woody-5',
]

REPO_ROOT = osp.dirname(osp.abspath(__file__))
RUNNER = osp.join(REPO_ROOT, 'radar_gp_state_estimation.py')
ACTIVE_CONFIG = osp.join(REPO_ROOT, 'config.yaml')


def set_scalar(config_text, key, value, required=True):
    """Replace the value of a top-level-indented `key: ...` line (not a commented-out one)."""
    pattern = re.compile(rf'^(\s*{re.escape(key)}\s*:\s*).*$', re.MULTILINE)
    new_text, n = pattern.subn(lambda m: m.group(1) + value, config_text, count=1)
    if n == 0 and required:
        raise SystemExit(f"Could not find an uncommented '{key}:' line in the base config to override.")
    return new_text


def build_config(base_text, data_path):
    text = base_text
    text = set_scalar(text, 'data_path', f"'{data_path}'")
    text = set_scalar(text, 'multi_sequence', 'false')
    return text


def seq_id_for(data_path):
    return osp.basename(data_path.rstrip('/'))


def already_done(seq_id):
    result_path = osp.join(REPO_ROOT, 'output', seq_id, 'odometry_result', f'{seq_id}.txt')
    return osp.exists(result_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default='config.yaml',
                         help='Base config file to reuse for every run (default: config.yaml).')
    parser.add_argument('--skip-existing', action='store_true',
                         help='Skip sequences that already have an output/<seq_id>/odometry_result/<seq_id>.txt.')
    parser.add_argument('--no-eval', dest='eval', action='store_false',
                         help='Skip running boreas_eval.py after all sequences finish (run by default).')
    args = parser.parse_args()

    sequences = [s if osp.isabs(s) else osp.join(DATA_ROOT, s) for s in SEQUENCES]
    if not sequences:
        parser.error('SEQUENCES is empty. Add sequence names at the top of this file.')

    seen = set()  # de-duplicate while preserving order
    unique_sequences = []
    for s in sequences:
        s = osp.abspath(s)
        if s not in seen:
            seen.add(s)
            unique_sequences.append(s)
    sequences = unique_sequences

    base_config_path = osp.abspath(args.config)
    if not osp.exists(base_config_path):
        raise SystemExit(f"Base config not found: {base_config_path}")
    with open(base_config_path) as f:
        base_text = f.read()

    # Preserve whatever is currently in config.yaml (the file the runner actually reads)
    original_active_text = None
    if osp.exists(ACTIVE_CONFIG):
        with open(ACTIVE_CONFIG) as f:
            original_active_text = f.read()

    os.makedirs(osp.join(REPO_ROOT, 'output'), exist_ok=True)
    results = []  # (seq_id, status, elapsed_seconds)

    try:
        for i, data_path in enumerate(sequences, 1):
            seq_id = seq_id_for(data_path)
            print(f"\n[{i}/{len(sequences)}] {seq_id}  ({data_path})")

            if args.skip_existing and already_done(seq_id):
                print("  -> skipped (already has odometry_result)")
                results.append((seq_id, 'skipped', 0.0))
                continue

            if not osp.isdir(data_path):
                print(f"  -> skipped (not a directory: {data_path})")
                results.append((seq_id, 'missing', 0.0))
                continue

            config_text = build_config(base_text, data_path)

            with open(ACTIVE_CONFIG, 'w') as f:
                f.write(config_text)

            log_path = osp.join(REPO_ROOT, 'output', f'{seq_id}_run.log')
            start = time.time()
            with open(log_path, 'w') as log_f:
                proc = subprocess.run(
                    [sys.executable, RUNNER],
                    cwd=REPO_ROOT,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                )
            elapsed = time.time() - start

            if proc.returncode == 0:
                print(f"  -> done in {elapsed:.1f}s (log: {log_path})")
                results.append((seq_id, 'ok', elapsed))
            else:
                print(f"  -> FAILED (exit {proc.returncode}) after {elapsed:.1f}s, see {log_path}")
                results.append((seq_id, 'failed', elapsed))
    finally:
        # Always restore whatever config.yaml originally held
        if original_active_text is not None:
            with open(ACTIVE_CONFIG, 'w') as f:
                f.write(original_active_text)
        elif osp.exists(ACTIVE_CONFIG):
            os.remove(ACTIVE_CONFIG)

    print("\n===== Batch summary =====")
    for seq_id, status, elapsed in results:
        suffix = f" ({elapsed:.1f}s)" if elapsed else ""
        print(f"  {status.upper():8s} {seq_id}{suffix}")
    n_ok = sum(1 for _, s, _ in results if s == 'ok')
    n_failed = sum(1 for _, s, _ in results if s == 'failed')
    print(f"{n_ok}/{len(results)} succeeded, {n_failed} failed.")

    if args.eval:
        print("\n===== Running boreas_eval.py =====")
        subprocess.run([sys.executable, osp.join(REPO_ROOT, 'boreas_eval.py'), 'output'], cwd=REPO_ROOT)


if __name__ == '__main__':
    main()
