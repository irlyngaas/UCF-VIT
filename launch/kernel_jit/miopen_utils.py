#!/usr/bin/env python3
"""MIOpen pre-compiled kernel cache helpers for production launchers.

Subcommands:
  path    config.yaml [--print key|dir|json]   cache key + Lustre seed directory
  dump    <dest_root>                          WAL-safe per-node cache snapshot
  publish <dump_dir> <seed_dir> --key KEY ...  install agreed seed.tar on Lustre
  verify  config.yaml [--expect-key KEY]       re-read a seed the way a sweep does

Seed layout: {root}/{cache-key}/seed.tar
Root from miopen_precompile.root in yaml, MIOPEN_CACHE_ROOT, or a default under
ven114/$USER.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import sqlite3
import sys
import tarfile
from pathlib import Path

import yaml

KEY_VERSION = 2
SQLITE_SUFFIXES = (".ukdb", ".udb", ".db")
SKIP_SUFFIXES = ("-shm", "-wal", "-journal")
BLOB_HASH_LEN = 16
SEED_SUBDIRS = ("kcache", "userdb")


# --- path (cache key + seed directory) ---------------------------------------


def default_seed_root() -> Path:
    user = os.environ.get("USER")
    if not user:
        raise SystemExit(
            "USER is unset; set MIOPEN_CACHE_ROOT or miopen_precompile.root in the config"
        )
    return Path(f"/lustre/orion/stf006/proj-shared/{user}/miopen_seed")


def expand_path(path_str: str) -> Path:
    return Path(os.path.expandvars(path_str))


def seed_root_from_config(conf: dict, cli_seed_root: Path | None = None) -> Path:
    if cli_seed_root is not None:
        return cli_seed_root
    # miopen_seed is the pre-rename key, still present in the older configs.
    section = conf.get("miopen_precompile") or conf.get("miopen_seed") or {}
    if root := section.get("root"):
        return expand_path(str(root))
    if env := os.environ.get("MIOPEN_CACHE_ROOT"):
        return expand_path(env)
    return default_seed_root()


def shape_fields(conf: dict) -> dict:
    model = conf.get("model", {}) or {}
    par = conf.get("parallelism", {}) or {}
    tiling = conf.get("tiling", {}) or {}
    ap = conf.get("ap", {}) or {}
    data = conf.get("data", {}) or {}
    dataset_options = conf.get("dataset_options", {}) or {}

    dict_root_dirs = data.get("dict_root_dirs", {}) or {}
    active = sorted(dict_root_dirs.keys())

    per_dataset = {}
    for k in active:
        per_dataset[k] = {
            "num_channels": (data.get("num_channels", {}) or {}).get(k),
            "in_vars": (data.get("dict_in_variables", {}) or {}).get(k, []),
            "out_vars": (data.get("dict_out_variables", {}) or {}).get(k, []),
        }

    do_tiling = bool(tiling.get("do_tiling", False))
    do_ap = bool(ap.get("do_ap", False))
    return {
        "key_version": KEY_VERSION,
        "dataset": data.get("dataset"),
        "img_size": data.get("img_size"),
        "twoD": data.get("twoD"),
        "patch_size": data.get("patch_size"),
        "resize": (dataset_options.get("resize", {}) or {}).get(data.get("dataset")),
        "batch_size_per_rank": (conf.get("dataloader", {}) or {}).get("batch_size"),
        "data_type": (conf.get("trainer", {}) or {}).get("data_type"),
        "model": {
            "type": model.get("type"),
            "embed_dim": model.get("embed_dim"),
            "depth": model.get("depth"),
            "num_heads": model.get("num_heads"),
            "mlp_ratio": model.get("mlp_ratio"),
            "use_channel_aggregation": model.get("use_channel_aggregation"),
            "linear_decoder": model.get("linear_decoder"),
            "decoder_embed_dim": model.get("decoder_embed_dim"),
            "decoder_depth": model.get("decoder_depth"),
            "decoder_num_heads": model.get("decoder_num_heads"),
            "decoder_mlp_ratio": model.get("decoder_mlp_ratio"),
            "num_classes": model.get("num_classes"),
            "num_time_steps": model.get("num_time_steps"),
            "mask_ratio": model.get("mask_ratio"),
        },
        "fsdp_size": par.get("fsdp_size"),
        "tensor_par_size": par.get("tensor_par_size"),
        "tiling": {
            "do_tiling": do_tiling,
            "div": tiling.get("div") if do_tiling else None,
            "tile_overlap": tiling.get("tile_overlap") if do_tiling else None,
        },
        "ap": {
            "do_ap": do_ap,
            "do_gpu_ap": ap.get("do_gpu_ap") if do_ap else None,
            "fixed_length": ap.get("fixed_length") if do_ap else None,
            "separate_channels": ap.get("separate_channels") if do_ap else None,
            "interp_size": ap.get("interp_size") if do_ap else None,
        },
        "datasets": per_dataset,
    }


def key_hash(fields: dict) -> str:
    blob = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def derive_tag(config_path: Path) -> str:
    return config_path.stem


def resolve_path(
    config_path: Path,
    seed_root: Path | None = None,
    tag: str | None = None,
):
    conf = yaml.safe_load(config_path.read_text())
    fields = shape_fields(conf)
    h = key_hash(fields)
    tag = tag or derive_tag(config_path)
    key = f"{tag}-{h}"
    root = seed_root_from_config(conf, seed_root)
    return fields, key, root / key


def cmd_path(args: argparse.Namespace) -> int:
    if not args.config.is_file():
        print(f"no such config: {args.config}", file=sys.stderr)
        return 1

    fields, key, seed_dir = resolve_path(args.config, args.seed_root, args.tag)
    if args.what == "key":
        print(key)
    elif args.what == "dir":
        print(seed_dir)
    else:
        print(json.dumps({"key": key, "seed_dir": str(seed_dir), "fields": fields}, indent=2))
    return 0


# --- dump (per-node cache snapshot) ------------------------------------------


def hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:BLOB_HASH_LEN]


def normalize_value(v) -> str:
    if isinstance(v, (bytes, bytearray)):
        return f"blob(len={len(v)},sha={hash_bytes(bytes(v))})"
    if v is None:
        return "NULL"
    return str(v)


def dump_sqlite(path: Path, out_lines: list) -> dict:
    stats = {}
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        for table in tables:
            cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')]
            keep = [c for c in cols if c.lower() != "id"]
            sel = ", ".join(f'"{c}"' for c in keep)
            rows = [
                " | ".join(normalize_value(v) for v in row)
                for row in conn.execute(f'SELECT {sel} FROM "{table}"')
            ]
            rows.sort()
            stats[table] = len(rows)
            out_lines.append(f"# TABLE {path.name}:{table} columns={keep} rows={len(rows)}")
            for row in rows:
                out_lines.append(f"{path.name}:{table} :: {row}")
    finally:
        conn.close()
    return stats


def snapshot_sqlite(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(src))
    try:
        conn.execute("VACUUM INTO ?", (str(dst),))
    except sqlite3.OperationalError:
        dest = sqlite3.connect(str(dst))
        try:
            conn.backup(dest)
        finally:
            dest.close()
    finally:
        conn.close()


def collect_dump(src_root: Path, label: str, dest: Path, manifest: list, summary: list) -> None:
    if not src_root or not src_root.is_dir():
        summary.append(f"{label}: MISSING ({src_root})")
        return

    files = sorted(p for p in src_root.rglob("*") if p.is_file())
    files = [p for p in files if not p.name.endswith(SKIP_SUFFIXES)]
    if not files:
        summary.append(f"{label}: EMPTY ({src_root})")
        return

    for src in files:
        rel = src.relative_to(src_root)
        out = dest / label / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix in SQLITE_SUFFIXES:
            snapshot_sqlite(src, out)
            stats = dump_sqlite(out, manifest)
            detail = ", ".join(f"{t}={n}" for t, n in sorted(stats.items()))
            summary.append(f"{label}/{rel}: sqlite {out.stat().st_size}B [{detail}]")
        else:
            shutil.copy2(src, out)
            if src.suffix in (".txt", ".log"):
                lines = sorted(out.read_text(errors="replace").splitlines())
                manifest.append(f"# TEXT {rel} lines={len(lines)}")
                manifest.extend(f"{src.name} :: {ln}" for ln in lines)
            summary.append(f"{label}/{rel}: file {out.stat().st_size}B")


def cmd_dump(args: argparse.Namespace) -> int:
    host = socket.gethostname()
    dest = args.dest_root / host
    dest.mkdir(parents=True, exist_ok=True)

    manifest: list = []
    summary: list = []

    for env, label in (
        ("MIOPEN_CUSTOM_CACHE_DIR", "kcache"),
        ("MIOPEN_USER_DB_PATH", "userdb"),
    ):
        raw = os.environ.get(env)
        summary.append(f"# {env}={raw}")
        collect_dump(Path(raw) if raw else None, label, dest, manifest, summary)

    (dest / "manifest.txt").write_text("\n".join(manifest) + "\n")

    header = [
        f"host={host}",
        f"nodes={os.environ.get('SLURM_JOB_NUM_NODES', '?')}",
        f"jobid={os.environ.get('SLURM_JOB_ID', '?')}",
        f"config={os.environ.get('CONFIG', '?')}",
    ]
    (dest / "summary.txt").write_text("\n".join(header + summary) + "\n")

    print(f"[miopen_utils dump] {host} -> {dest}")
    for line in header + summary:
        print(f"  {line}")
    return 0


# --- publish (install seed.tar on Lustre) ------------------------------------


# A find-db entry is <problem>=<Solver>:<ms>,<workspace>,<algo>;<Solver2>:...
# The solver picked, its workspace and its algo have to match across nodes. The
# milliseconds are a benchmark measured on that node's own GPU and never match,
# so hashing them made any build on more than one node impossible to publish
# (job 5238259: identical kernel blobs, 8/8 SHAs equal, refused on timing noise).
FINDDB_TIMING_RE = re.compile(r":\s*\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\s*,")


def normalize_manifest(text: str) -> str:
    return "\n".join(
        FINDDB_TIMING_RE.sub(":,", ln) if ".ufdb.txt ::" in ln else ln
        for ln in text.splitlines()
    )


def manifest_fp(path: Path) -> str:
    body = normalize_manifest(path.read_text(errors="replace"))
    return hashlib.sha256(body.encode()).hexdigest()[:12]


def manifest_stats(path: Path) -> tuple[int, int]:
    kernels = finddb = 0
    for ln in path.read_text(errors="replace").splitlines():
        if ":kern_db ::" in ln:
            kernels += 1
        elif ".ufdb.txt ::" in ln:
            finddb += 1
    return kernels, finddb


def load_manifests(dump_dir: Path) -> list[tuple[str, Path]]:
    out = []
    for path in sorted(dump_dir.glob("*/manifest.txt")):
        out.append((path.parent.name, path))
    if not out:
        raise SystemExit(f"no per-node manifests under {dump_dir}")
    return out


def check_agreement(manifests: list[tuple[str, Path]]) -> None:
    fps = {manifest_fp(p): (host, p) for host, p in manifests}
    if len(fps) == 1:
        host, p = next(iter(fps.values()))
        print(f"  all {len(manifests)} node(s) agree (fp={manifest_fp(p)}, host={host})")
        return
    print("  REFUSING to publish: nodes disagree", file=sys.stderr)
    for host, p in manifests:
        k, f = manifest_stats(p)
        print(f"    {host}: fp={manifest_fp(p)} kernels={k} finddb={f}", file=sys.stderr)

    # Naming the differing lines is the whole point of the check; without them
    # the only way to tell a real cache divergence from noise is to go read the
    # dumps by hand.
    (ha, pa), (hb, pb) = manifests[0], manifests[1]
    la = normalize_manifest(pa.read_text(errors="replace")).splitlines()
    lb = normalize_manifest(pb.read_text(errors="replace")).splitlines()
    print(f"  first differences, {ha} vs {hb}:", file=sys.stderr)
    shown = 0
    for i in range(max(len(la), len(lb))):
        a = la[i] if i < len(la) else "<missing>"
        b = lb[i] if i < len(lb) else "<missing>"
        if a != b:
            print(f"    line {i + 1}:", file=sys.stderr)
            print(f"      {ha}: {a[:160]}", file=sys.stderr)
            print(f"      {hb}: {b[:160]}", file=sys.stderr)
            shown += 1
            if shown == 3:
                break
    raise SystemExit(2)


def cmd_publish(args: argparse.Namespace) -> int:
    print(f"publishing seed for key={args.key}")
    manifests = load_manifests(args.dump_dir)
    check_agreement(manifests)

    ref_host, ref_manifest = manifests[0]
    src_root = ref_manifest.parent
    if not any((src_root / s).is_dir() for s in SEED_SUBDIRS):
        raise SystemExit(f"no kcache/userdb under {src_root}")

    seed = args.seed_dir
    tree = seed / "tree"
    if tree.exists():
        shutil.rmtree(tree)
    tree.mkdir(parents=True, exist_ok=True)

    members = []
    for sub in SEED_SUBDIRS:
        srcs = src_root / sub
        if not srcs.is_dir():
            print(f"  note: no {sub}/ in dump (nothing to seed for it)")
            continue
        shutil.copytree(srcs, tree / sub)
        members.append(sub)

    tar_path = seed / "seed.tar"
    with tarfile.open(tar_path, "w") as tf:
        for sub in members:
            tf.add(tree / sub, arcname=sub)

    shutil.copy2(ref_manifest, seed / "manifest.txt")

    if args.key_json and args.key_json.is_file():
        shutil.copy2(args.key_json, seed / "key.json")

    n_kernels, n_finddb = manifest_stats(ref_manifest)
    files = sorted(p for p in tree.rglob("*") if p.is_file())
    (seed / "provenance.txt").write_text(
        "\n".join(
            [
                f"key={args.key}",
                f"config={args.config}",
                f"built_by_jobid={args.jobid}",
                f"built_on_nodes={args.nodes}",
                f"hosts_agreeing={','.join(h for h, _ in manifests)}",
                f"reference_host={ref_host}",
                f"kernels={n_kernels}",
                f"finddb_problems={n_finddb}",
                f"manifest_fp={manifest_fp(ref_manifest)}",
                f"seed_tar_bytes={tar_path.stat().st_size}",
                "files:",
            ]
            + [f"  {p.relative_to(tree)} ({p.stat().st_size}B)" for p in files]
        )
        + "\n"
    )

    print(f"  installed {len(files)} file(s) -> {seed}")
    print(f"  seed.tar = {tar_path.stat().st_size}B")
    print(f"  kernels={n_kernels} finddb_problems={n_finddb}")
    return 0


# --- verify (read a seed back the way a consumer resolves it) -----------------


PROVENANCE_FILE_RE = re.compile(r"^ {2}(\S+) \((\d+)B\)$")


def read_provenance(path: Path) -> tuple[str | None, int | None, dict[str, int]]:
    key = None
    tar_bytes = None
    files: dict[str, int] = {}
    for ln in path.read_text(errors="replace").splitlines():
        if ln.startswith("key="):
            key = ln.split("=", 1)[1]
        elif ln.startswith("seed_tar_bytes="):
            tar_bytes = int(ln.split("=", 1)[1])
        elif m := PROVENANCE_FILE_RE.match(ln):
            files[m.group(1)] = int(m.group(2))
    return key, tar_bytes, files


def cmd_verify(args: argparse.Namespace) -> int:
    # The builder knows the seed directory as a shell variable, but every
    # consumer re-derives it from the config. Anything that makes those two
    # disagree -- a relative miopen_precompile.root, a KEY_VERSION bump, a
    # renamed config -- publishes a seed nothing ever finds again, so this
    # starts from the config rather than taking the directory as an argument.
    if not args.config.is_file():
        print(f"no such config: {args.config}", file=sys.stderr)
        return 1

    _, key, seed_dir = resolve_path(args.config, args.seed_root, args.tag)
    print(f"  key      = {key}")
    print(f"  seed_dir = {seed_dir}")

    problems: list[str] = []
    if args.expect_key and key != args.expect_key:
        problems.append(
            f"key drift: published as {args.expect_key}, "
            f"but this config now resolves to {key}"
        )

    prov_path = seed_dir / "provenance.txt"
    want_files: dict[str, int] = {}
    want_bytes: int | None = None
    if prov_path.is_file():
        prov_key, want_bytes, want_files = read_provenance(prov_path)
        if prov_key and prov_key != key:
            problems.append(f"provenance records key={prov_key}, config resolves to {key}")
    else:
        problems.append(f"no provenance.txt at {prov_path}")

    tar_path = seed_dir / "seed.tar"
    if not tar_path.is_file():
        problems.append(f"no seed.tar at {tar_path}")
    else:
        size = tar_path.stat().st_size
        print(f"  seed.tar = {size}B")
        if want_bytes is not None and want_bytes != size:
            problems.append(f"seed.tar is {size}B, provenance recorded {want_bytes}B")

        got: dict[str, int] | None = None
        if size == 0:
            # tarfile's own error for an empty file is four lines of failed
            # decompression guesses, which buries the one fact that matters.
            problems.append(f"{tar_path} is empty")
        else:
            try:
                with tarfile.open(tar_path) as tf:
                    got = {m.name: m.size for m in tf.getmembers() if m.isfile()}
            except (tarfile.TarError, OSError) as exc:
                problems.append(f"cannot read {tar_path}: {exc}")

        if got is not None:
            print(f"  members  = {len(got)}")
            if not got:
                problems.append(f"{tar_path} holds no files")
            for name, want in sorted(want_files.items()):
                if name not in got:
                    problems.append(f"seed.tar is missing {name}")
                elif got[name] != want:
                    problems.append(
                        f"seed.tar {name} is {got[name]}B, provenance recorded {want}B"
                    )
            for name in sorted(set(got) - set(want_files)):
                problems.append(f"seed.tar holds {name}, which provenance does not list")

    if problems:
        # Both streams land in one batch log, so without this the failure list
        # prints above the key and path it refers to.
        sys.stdout.flush()
        print(f"  VERIFY FAILED for {key}", file=sys.stderr)
        for p in problems:
            print(f"    {p}", file=sys.stderr)
        return 1

    print("  verify OK: a sweep resolving this config will find this seed")
    return 0


# --- CLI ---------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    path_p = sub.add_parser("path", help="resolve cache key and seed directory from a config")
    path_p.add_argument("config", type=Path)
    path_p.add_argument(
        "--seed-root",
        type=Path,
        default=None,
        help="override miopen_precompile.root / MIOPEN_CACHE_ROOT / default",
    )
    path_p.add_argument("--tag", default=None)
    path_p.add_argument("--print", dest="what", choices=("key", "dir", "json"), default="key")
    path_p.set_defaults(func=cmd_path)

    dump_p = sub.add_parser("dump", help="snapshot this node's MIOpen caches to Lustre")
    dump_p.add_argument("dest_root", type=Path)
    dump_p.set_defaults(func=cmd_dump)

    pub_p = sub.add_parser("publish", help="install agreed per-node dumps as seed.tar")
    pub_p.add_argument("dump_dir", type=Path)
    pub_p.add_argument("seed_dir", type=Path)
    pub_p.add_argument("--key", required=True)
    pub_p.add_argument("--key-json", type=Path, help="output of: miopen_utils.py path CFG --print json")
    pub_p.add_argument("--config", default="")
    pub_p.add_argument("--jobid", default="")
    pub_p.add_argument("--nodes", default="")
    pub_p.set_defaults(func=cmd_publish)

    ver_p = sub.add_parser("verify", help="check a published seed is findable and readable")
    ver_p.add_argument("config", type=Path)
    ver_p.add_argument("--seed-root", type=Path, default=None)
    ver_p.add_argument("--tag", default=None)
    ver_p.add_argument("--expect-key", default="", help="key the seed was published under")
    ver_p.set_defaults(func=cmd_verify)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
