#!/usr/bin/env python3
"""Extract and plot global, rank, and modality epoch losses from run logs.

Examples:
    ./dev_scripts/plot_training_losses.py /path/to/run/logs
    ./dev_scripts/plot_training_losses.py '/path/to/logs/*.out' -o losses.png
    ./dev_scripts/plot_training_losses.py /path/to/logs --keep-replays

The default output includes a PNG and a long-form CSV. Replayed or rolled-back
epochs are resolved by following the chronological allocation lineage; pass
``--keep-replays`` to retain every historical attempt instead.
"""

import argparse
import ast
import csv
import glob
import math
import re
from dataclasses import dataclass
from pathlib import Path


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
RANK_PREFIX_RE = re.compile(r"\[rank(?P<rank>\d+)\]:")
RANK_TEXT_RE = re.compile(r"\bworld_rank\s*[=:]?\s*(?P<rank>\d+)\b")
RANK_FILE_RE = re.compile(r"rank[_-]?(?P<rank>\d+)", re.IGNORECASE)
OVERALL_RE = re.compile(
    rf"epoch:\s*(?P<epoch>\d+)\s+epoch_loss\s+"
    rf"(?:tensor\()?\s*(?P<loss>{FLOAT})"
)
MODALITY_RE = re.compile(
    r"epoch:\s*(?P<epoch>\d+)\s+modality_losses\s+"
    r"\((?P<status>complete|partial)\)\s+(?P<values>\{.*\})"
)
START_RE = re.compile(r"Starting epoch\s+(?P<epoch>\d+)")


@dataclass(frozen=True)
class LossRecord:
    epoch: int
    loss: float
    scope: str
    rank: int | None
    modality: str | None
    source: str
    line: int
    occurrence: int


def _rank_for_line(text, path):
    for pattern in (RANK_PREFIX_RE, RANK_TEXT_RE):
        match = pattern.search(text)
        if match:
            return int(match.group('rank'))
    match = RANK_FILE_RE.search(path.name)
    return int(match.group('rank')) if match else None


def parse_log(path, occurrence_start=0):
    """Return completed epoch-loss records from one log file."""
    path = Path(path)
    records = []
    occurrence = occurrence_start
    with path.open('r', errors='replace') as stream:
        for line_number, text in enumerate(stream, 1):
            rank = _rank_for_line(text, path)
            overall = OVERALL_RE.search(text)
            if overall:
                loss = float(overall.group('loss'))
                if math.isfinite(loss):
                    occurrence += 1
                    records.append(LossRecord(
                        epoch=int(overall.group('epoch')),
                        loss=loss,
                        scope='overall',
                        rank=rank,
                        modality=None,
                        source=str(path),
                        line=line_number,
                        occurrence=occurrence,
                    ))

            modality = MODALITY_RE.search(text)
            if not modality or modality.group('status') != 'complete':
                continue
            try:
                values = ast.literal_eval(modality.group('values'))
            except (SyntaxError, ValueError):
                continue
            if not isinstance(values, dict):
                continue
            for name, value in values.items():
                try:
                    loss = float(value)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(loss):
                    continue
                occurrence += 1
                records.append(LossRecord(
                    epoch=int(modality.group('epoch')),
                    loss=loss,
                    scope='modality',
                    rank=rank,
                    modality=str(name),
                    source=str(path),
                    line=line_number,
                    occurrence=occurrence,
                ))
    return records


def first_started_epoch(path):
    """Return the first epoch attempted by an allocation, if it is logged."""
    with Path(path).open('r', errors='replace') as stream:
        for text in stream:
            match = START_RE.search(text)
            if match:
                return int(match.group('epoch'))
    return None


def discover_logs(inputs):
    """Expand files, directories, and shell-style patterns deterministically."""
    paths = set()
    for item in inputs:
        matches = glob.glob(item, recursive=True)
        candidates = [Path(match) for match in matches] or [Path(item)]
        for candidate in candidates:
            if candidate.is_dir():
                paths.update(candidate.rglob('*.out'))
            elif candidate.is_file():
                paths.add(candidate)
    return sorted(paths, key=lambda path: (path.stat().st_mtime_ns, str(path)))


def collect_records(paths, keep_replays=False):
    records = []
    retained = {}
    occurrence = 0
    for path in paths:
        parsed = parse_log(path, occurrence)
        records.extend(parsed)
        if parsed:
            occurrence = parsed[-1].occurrence
        if keep_replays:
            continue

        # Follow the lineage that was actually continued. If a later Slurm
        # allocation starts at an earlier/same epoch, all retained observations
        # from that epoch onward belonged to the superseded branch. This also
        # handles intentional numerical rollback and interrupted-epoch replay.
        restart_epoch = first_started_epoch(path)
        if restart_epoch is not None:
            retained = {
                key: value for key, value in retained.items()
                if value.epoch < restart_epoch
            }
        for record in parsed:
            key = (record.scope, record.rank, record.modality, record.epoch)
            retained[key] = record
    if keep_replays:
        return records
    return sorted(retained.values(), key=lambda item: item.occurrence)


def write_csv(records, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow([
            'epoch', 'loss', 'scope', 'rank', 'modality',
            'source', 'line', 'occurrence',
        ])
        for record in sorted(records, key=lambda item: (
                item.epoch, item.scope, item.rank if item.rank is not None else -1,
                item.modality or '', item.occurrence)):
            writer.writerow([
                record.epoch, f'{record.loss:.12g}', record.scope,
                '' if record.rank is None else record.rank,
                record.modality or '', record.source, record.line,
                record.occurrence,
            ])


def _series(records, scope):
    grouped = {}
    for record in records:
        if record.scope != scope:
            continue
        if scope == 'overall':
            label = 'global' if record.rank is None else f'rank {record.rank}'
        else:
            label = record.modality or 'unknown'
            if record.rank is not None:
                label = f'{label} (rank {record.rank})'
        grouped.setdefault(label, []).append(record)
    return grouped


def plot_records(records, output_path, title=None, log_y=False,
                 robust_percentiles=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    panels = []
    for scope, panel_title in (
        ('overall', 'Overall epoch loss'),
        ('modality', 'Per-modality epoch loss'),
    ):
        grouped = _series(records, scope)
        if grouped:
            panels.append((panel_title, grouped))
    if not panels:
        raise ValueError('No completed epoch-loss records were found')

    fig, axes = plt.subplots(
        len(panels), 1, figsize=(10, 4.5 * len(panels)), squeeze=False
    )
    for ax, (panel_title, grouped) in zip(axes[:, 0], panels):
        panel_values = []
        for label, series in sorted(grouped.items()):
            ordered = sorted(series, key=lambda item: (item.epoch, item.occurrence))
            epochs = [item.epoch for item in ordered]
            losses = [item.loss for item in ordered]
            panel_values.extend(losses)
            ax.plot(epochs, losses, '-o', markersize=3, linewidth=1.4,
                    label=label)
        if log_y and all(value > 0 for value in panel_values):
            ax.set_yscale('log')
        if robust_percentiles and len(panel_values) >= 4:
            low, high = np.percentile(panel_values, robust_percentiles)
            if high > low:
                padding = (high - low) * 0.08
                ax.set_ylim(low - padding, high + padding)
        elif len(panel_values) == 1:
            value = panel_values[0]
            padding = max(abs(value) * 0.1, 1e-6)
            ax.set_ylim(value - padding, value + padding)
        ax.set_title(panel_title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE loss')
        ax.grid(True, alpha=0.25)
        ax.legend(loc='best')
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def print_summary(records):
    for scope in ('overall', 'modality'):
        for label, series in sorted(_series(records, scope).items()):
            ordered = sorted(series, key=lambda item: item.epoch)
            best = min(ordered, key=lambda item: item.loss)
            first, last = ordered[0], ordered[-1]
            print(
                f'{scope}:{label}: epochs {first.epoch}-{last.epoch}, '
                f'loss {first.loss:.6g}->{last.loss:.6g}, '
                f'best {best.loss:.6g} at epoch {best.epoch}'
            )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'inputs', nargs='+', help='Log files, directories, or glob patterns'
    )
    parser.add_argument('-o', '--output', default='training_losses.png')
    parser.add_argument('--csv', dest='csv_path', default=None)
    parser.add_argument('--title', default=None)
    parser.add_argument('--keep-replays', action='store_true')
    parser.add_argument('--log-y', action='store_true')
    parser.add_argument(
        '--robust-percentiles', nargs=2, type=float, metavar=('LOW', 'HIGH'),
        help='Optional y-axis clipping percentiles, for example 2 98',
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    paths = discover_logs(args.inputs)
    if not paths:
        raise SystemExit('No input log files found')
    if args.robust_percentiles:
        low, high = args.robust_percentiles
        if not 0 <= low < high <= 100:
            raise SystemExit('Percentiles must satisfy 0 <= LOW < HIGH <= 100')
    records = collect_records(paths, keep_replays=args.keep_replays)
    if not records:
        raise SystemExit('No completed epoch-loss records found')
    csv_path = args.csv_path or str(Path(args.output).with_suffix('.csv'))
    write_csv(records, csv_path)
    plot_records(
        records, args.output, title=args.title, log_y=args.log_y,
        robust_percentiles=args.robust_percentiles,
    )
    print(f'Parsed {len(paths)} logs and {len(records)} retained records')
    print(f'Plot: {args.output}')
    print(f'CSV: {csv_path}')
    print_summary(records)


if __name__ == '__main__':
    main()
