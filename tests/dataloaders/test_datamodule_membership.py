"""Tests for NativePytorchDataModule's train/val/test membership stability.

Per-epoch file-order reshuffling (now done inside FileReader.__iter__ itself,
see its own epoch_shuffle_seed docstring entry) is a legitimate thing to do --
training variety and per-rank sharding fairness -- but it used to happen
*before* dict_start_idx/dict_end_idx's ratio slice was applied (inside
FileReader, constructed from set_iterative_dataloader, back when
NativePytorchDataModule.setup()/reset() did the reshuffling in the main
process instead). That was harmless while dict_start_idx/dict_end_idx was
always the full [0,1) range (every shipped config, before auto train/val/test
splitting existed): shuffling before taking 100% of a list is a no-op either
way. Once it could be a genuine partial range (auto-split's whole point), it
became a real data-leakage bug: which files counted as "train" vs held-out
"val"/"test" would silently change on every checkpoint restart, and even
between epochs of the *same* run, since reset() ran after every epoch.

Fixed by slicing once, deterministically (sorted() first, since
os.listdir/glob.glob/FileLister order isn't guaranteed stable either), in
__init__ -- before any per-epoch reshuffle ever sees the list. These tests
construct NativePytorchDataModule directly against a small real fake
directory (no mocking) and check dict_lister_trains -- the fixed membership
set by __init__, never mutated by setup() or by FileReader's own internal
reshuffling (which works from its own local copy, never touching
self.dict_lister_trains) -- rather than exercising the full dataloader
pipeline.

Immediate follow-up, imagenet specifically: process_root_dirs used to bucket
imagenet's classes into data_par_size buckets *before* any slicing happened,
so the split above wasn't actually data_par_size-independent for imagenet --
a checkpoint restart with a different node count (or just running val.py/
test.py at a different parallelism than the training run being evaluated)
could still reshuffle which images count as train/val/test. Fixed by moving
bucketing (via the new bucket_file_list) to *after* the slice, operating on
already-sorted, already-resolved membership -- process_root_dirs no longer
buckets at all, just lists every image in deterministic order (see its own
docstring). test_imagenet_membership_deterministic_across_different_data_par_size
below is the direct regression test for the actual scenario asked about.
"""

from UCF_VIT.dataloaders.datamodule import NativePytorchDataModule
from UCF_VIT.dataloaders.dataset import FileReader


def _make_basic_ct_dir(tmp_path, num_files):
    root = tmp_path / "ct_root"
    images_tr = root / "imagesTr"
    images_tr.mkdir(parents=True)
    for i in range(num_files):
        (images_tr / f"img{i:03d}.nii.gz").write_text("")
    return str(root)


def _make_module(root, start_idx, end_idx):
    return NativePytorchDataModule(
        dict_root_dirs={"ct1": root},
        dict_start_idx={"ct1": start_idx},
        dict_end_idx={"ct1": end_idx},
        dict_buffer_sizes={"ct1": 10},
        dict_in_variables={"ct1": ["ct_res1"]},
        num_channels_used={"ct1": 1},
        data_par_size=1,
        dataset="basic_ct",
    )


def test_membership_deterministic_across_separate_constructions(tmp_path):
    # Simulates two separate process launches against the same data (e.g. a
    # checkpoint restart, or train.py and val.py run back to back) -- must
    # resolve to the exact same file set, not a freshly randomized one.
    root = _make_basic_ct_dir(tmp_path, num_files=20)

    first = _make_module(root, 0.0, 0.8)
    second = _make_module(root, 0.0, 0.8)

    assert first.dict_lister_trains["ct1"] == second.dict_lister_trains["ct1"]


def test_train_val_test_splits_are_disjoint_and_cover_everything(tmp_path):
    root = _make_basic_ct_dir(tmp_path, num_files=20)

    train = _make_module(root, 0.0, 0.8)
    val = _make_module(root, 0.8, 0.9)
    test = _make_module(root, 0.9, 1.0)

    train_files = set(train.dict_lister_trains["ct1"])
    val_files = set(val.dict_lister_trains["ct1"])
    test_files = set(test.dict_lister_trains["ct1"])

    assert len(train_files) == 16
    assert len(val_files) == 2
    assert len(test_files) == 2
    assert train_files.isdisjoint(val_files)
    assert train_files.isdisjoint(test_files)
    assert val_files.isdisjoint(test_files)
    all_files = {f"img{i:03d}.nii.gz" for i in range(20)}
    resolved_names = {f.split("/")[-1] for f in (train_files | val_files | test_files)}
    assert resolved_names == all_files


def test_membership_unaffected_by_setup(tmp_path):
    # setup() must never mutate dict_lister_trains itself -- FileReader does
    # its own per-epoch reshuffling internally now (see epoch_shuffle_seed),
    # from its own local copy.
    root = _make_basic_ct_dir(tmp_path, num_files=20)
    module = _make_module(root, 0.0, 0.8)
    before = list(module.dict_lister_trains["ct1"])

    module.batches_per_rank_epoch = {"ct1": 5}
    module.setup()
    after_setup = list(module.dict_lister_trains["ct1"])

    assert before == after_setup


def test_setup_only_builds_this_ranks_own_key_not_every_bucket(tmp_path):
    """The actual cost fix: setup() used to rebuild every key in
    dict_lister_trains (all data_par_size imagenet buckets) on every rank,
    even though train_dataloader only ever consumes one -- O(data_par_size)
    wasted shuffling/pipeline-construction per rank. Now it only ever builds
    this rank's own key. dict_lister_trains itself (built once, in __init__)
    still legitimately holds every bucket -- only dict_data_train (built by
    setup()) is restricted.
    """
    root = _make_imagenet_dir(tmp_path, num_classes=20, images_per_class=5)  # 100 images
    module = _make_imagenet_module(root, 0.0, 1.0, data_par_size=8)
    assert len(module.dict_lister_trains) == 8  # every bucket, as before

    module.batches_per_rank_epoch = {"imagenet": 5}
    module.setup()
    assert len(module.dict_data_train) == 1  # only this rank's own key


def _make_imagenet_dir(tmp_path, num_classes, images_per_class):
    root = tmp_path / "imagenet_root"
    for c in range(num_classes):
        cdir = root / f"class{c:03d}"
        cdir.mkdir(parents=True)
        for i in range(images_per_class):
            (cdir / f"img{i}.JPEG").write_text("")
    return str(root)


def _make_imagenet_module(root, start_idx, end_idx, data_par_size, bucket_shuffle_seed=None, dataset_group_list=''):
    return NativePytorchDataModule(
        dict_root_dirs={"imagenet": root},
        dict_start_idx={"imagenet": start_idx},
        dict_end_idx={"imagenet": end_idx},
        dict_buffer_sizes={"imagenet": 10},
        dict_in_variables={"imagenet": ["red", "green", "blue"]},
        num_channels_used={"imagenet": 3},
        data_par_size=data_par_size,
        dataset="imagenet",
        bucket_shuffle_seed=bucket_shuffle_seed,
        dataset_group_list=dataset_group_list,
    )


def _all_imagenet_files(module):
    return {f for bucket in module.dict_lister_trains.values() for f in bucket}


def test_imagenet_membership_deterministic_across_different_data_par_size(tmp_path):
    """The actual scenario raised: a checkpoint restart with fewer/more nodes
    (data_par_size changes) must not change which images count as train --
    only how those images get divided into per-rank buckets should depend on
    data_par_size, not which images are in the split at all.
    """
    root = _make_imagenet_dir(tmp_path, num_classes=20, images_per_class=5)

    many_ranks = _make_imagenet_module(root, 0.0, 0.8, data_par_size=100)
    few_ranks = _make_imagenet_module(root, 0.0, 0.8, data_par_size=4)

    # Bucket *structure* legitimately differs...
    assert set(many_ranks.dict_lister_trains.keys()) != set(few_ranks.dict_lister_trains.keys()) or len(many_ranks.dict_lister_trains) != len(few_ranks.dict_lister_trains)
    # ...but the *set* of images in the split is identical either way.
    assert _all_imagenet_files(many_ranks) == _all_imagenet_files(few_ranks)


def test_imagenet_train_val_test_splits_are_disjoint_and_cover_everything(tmp_path):
    root = _make_imagenet_dir(tmp_path, num_classes=20, images_per_class=5)  # 100 images

    train = _make_imagenet_module(root, 0.0, 0.8, data_par_size=8)
    val = _make_imagenet_module(root, 0.8, 0.9, data_par_size=8)
    test = _make_imagenet_module(root, 0.9, 1.0, data_par_size=8)

    train_files = _all_imagenet_files(train)
    val_files = _all_imagenet_files(val)
    test_files = _all_imagenet_files(test)

    assert len(train_files) == 80
    assert len(val_files) == 10
    assert len(test_files) == 10
    assert train_files.isdisjoint(val_files)
    assert train_files.isdisjoint(test_files)
    assert val_files.isdisjoint(test_files)


def test_imagenet_bucket_count_matches_data_par_size(tmp_path):
    root = _make_imagenet_dir(tmp_path, num_classes=20, images_per_class=5)  # 100 images
    module = _make_imagenet_module(root, 0.0, 1.0, data_par_size=8)

    assert set(module.dict_lister_trains.keys()) == set(range(8))


def _classes_in_bucket(bucket):
    return {f.split("/")[-2] for f in bucket}


def test_imagenet_without_shuffle_seed_each_bucket_is_a_narrow_class_range(tmp_path):
    # Documents the problem bucket_shuffle_seed fixes: without it, images
    # arrive sorted class-by-class (process_root_dirs) and bucketing is a
    # contiguous split, so each bucket only ever contains a handful of
    # adjacent classes.
    root = _make_imagenet_dir(tmp_path, num_classes=20, images_per_class=5)  # 100 images
    module = _make_imagenet_module(root, 0.0, 1.0, data_par_size=10, bucket_shuffle_seed=None)

    for bucket in module.dict_lister_trains.values():
        assert len(_classes_in_bucket(bucket)) <= 2  # 20 classes / 10 buckets = 2 classes/bucket


def test_imagenet_bucket_shuffle_seed_spreads_classes_across_buckets(tmp_path):
    root = _make_imagenet_dir(tmp_path, num_classes=20, images_per_class=5)  # 100 images
    module = _make_imagenet_module(root, 0.0, 1.0, data_par_size=10, bucket_shuffle_seed=42)

    # At least one bucket now contains images from more than the 2 adjacent
    # classes it would be limited to without shuffling.
    assert any(len(_classes_in_bucket(bucket)) > 2 for bucket in module.dict_lister_trains.values())


def test_imagenet_bucket_shuffle_seed_deterministic_across_separate_constructions(tmp_path):
    root = _make_imagenet_dir(tmp_path, num_classes=20, images_per_class=5)

    first = _make_imagenet_module(root, 0.0, 0.8, data_par_size=8, bucket_shuffle_seed=42)
    second = _make_imagenet_module(root, 0.0, 0.8, data_par_size=8, bucket_shuffle_seed=42)

    assert first.dict_lister_trains == second.dict_lister_trains


def test_imagenet_membership_stable_across_data_par_size_even_with_shuffle_seed(tmp_path):
    # The membership fix (previous test above) must keep holding once
    # shuffling is layered on top of it -- shuffling only affects which
    # bucket an image lands in, never whether it's in the split at all.
    root = _make_imagenet_dir(tmp_path, num_classes=20, images_per_class=5)

    many_ranks = _make_imagenet_module(root, 0.0, 0.8, data_par_size=100, bucket_shuffle_seed=42)
    few_ranks = _make_imagenet_module(root, 0.0, 0.8, data_par_size=4, bucket_shuffle_seed=42)

    assert _all_imagenet_files(many_ranks) == _all_imagenet_files(few_ranks)


def test_train_dataloader_gives_different_ranks_different_buckets(tmp_path, monkeypatch):
    """The highest-risk part of the setup() cost fix: train_dataloader
    used to independently recompute group_id (via gx) and search
    dict_data_train for the matching enumeration index, since dict_data_train
    held every bucket. Now dict_data_train only ever holds *this* rank's own
    bucket (built by setup(), via the same group_id logic, now
    factored into _my_dataset_key), so train_dataloader just takes its one
    entry directly. Simulates 4 separate ranks (monkeypatching
    torch.distributed.get_rank, matching test_dataset.py's own established
    pattern for this) with dataset_group_list="1:1:1:1" (matching what
    calculate_load_balancing_on_the_fly really produces for imagenet -- one
    rank per bucket, not the single-group default this module falls back to
    when dataset_group_list is omitted) and confirms each rank's DataLoader
    wraps a genuinely different bucket, covering every bucket between them.
    """
    root = _make_imagenet_dir(tmp_path, num_classes=20, images_per_class=5)  # 100 images
    data_par_size = 4

    seen_dict_keys = []
    for ddp_rank in range(data_par_size):
        monkeypatch.setattr("torch.distributed.get_rank", lambda ddp_rank=ddp_rank: ddp_rank)
        module = _make_imagenet_module(
            root, 0.0, 1.0, data_par_size=data_par_size, dataset_group_list="1:1:1:1",
        )
        module.batches_per_rank_epoch = {"imagenet": 5}
        module.setup()
        dataloader = module.train_dataloader()
        # collate_fn's dict_key= kwarg (functools.partial) records which
        # bucket this rank's DataLoader actually wraps.
        seen_dict_keys.append(dataloader.collate_fn.keywords["dict_key"])

    assert sorted(seen_dict_keys) == list(range(data_par_size))  # every bucket, each exactly once


# ---------------------------------------------------------------------------
# train_dataloader()'s persistent_workers + epoch_shuffle_seed threading --
# what lets a single train_dataloader() call be reused for a whole run
# instead of being rebuilt (via the now-removed reset()) every epoch. See
# training_scripts/train.py's own comment on why the DataLoader is built,
# and its worker pool forked, before this process's first CUDA context.
# ---------------------------------------------------------------------------


def test_train_dataloader_sets_persistent_workers_only_when_num_workers_positive(tmp_path):
    root = _make_basic_ct_dir(tmp_path, num_files=20)

    no_workers = NativePytorchDataModule(
        dict_root_dirs={"ct1": root}, dict_start_idx={"ct1": 0.0}, dict_end_idx={"ct1": 1.0},
        dict_buffer_sizes={"ct1": 10}, dict_in_variables={"ct1": ["ct_res1"]},
        num_channels_used={"ct1": 1}, data_par_size=1, dataset="basic_ct", num_workers=0,
    )
    no_workers.batches_per_rank_epoch = {"ct1": 5}
    no_workers.setup()
    assert no_workers.train_dataloader().persistent_workers is False

    with_workers = NativePytorchDataModule(
        dict_root_dirs={"ct1": root}, dict_start_idx={"ct1": 0.0}, dict_end_idx={"ct1": 1.0},
        dict_buffer_sizes={"ct1": 10}, dict_in_variables={"ct1": ["ct_res1"]},
        num_channels_used={"ct1": 1}, data_par_size=1, dataset="basic_ct", num_workers=2,
    )
    with_workers.batches_per_rank_epoch = {"ct1": 5}
    with_workers.setup()
    assert with_workers.train_dataloader().persistent_workers is True


def test_train_dataloader_threads_epoch_shuffle_seed_to_filereader(tmp_path, monkeypatch):
    """set_iterative_dataloader must pass epoch_shuffle_seed through to the
    FileReader it builds -- the actual mechanism that lets FileReader
    reshuffle per-epoch on its own (see its own docstring), now that setup()
    no longer shuffles anything itself.
    """
    root = _make_basic_ct_dir(tmp_path, num_files=20)
    module = NativePytorchDataModule(
        dict_root_dirs={"ct1": root}, dict_start_idx={"ct1": 0.0}, dict_end_idx={"ct1": 1.0},
        dict_buffer_sizes={"ct1": 10}, dict_in_variables={"ct1": ["ct_res1"]},
        num_channels_used={"ct1": 1}, data_par_size=1, dataset="basic_ct",
        epoch_shuffle_seed=123,
    )
    module.batches_per_rank_epoch = {"ct1": 5}
    module.setup()

    k = next(iter(module.dict_data_train))
    # dict_data_train[k] is ProcessChannels(ShuffleIterableDataset(TileDataIter(FileReader(...))))
    file_reader = module.dict_data_train[k].dataset.dataset.dataset
    assert isinstance(file_reader, FileReader)
    assert file_reader.epoch_shuffle_seed == 123


def test_setup_no_longer_shuffles_dict_lister_trains_order_itself(tmp_path):
    """setup() passes dict_lister_trains[k] straight through to
    set_iterative_dataloader now (no _shuffle_and_replicate call of its own
    -- FileReader handles that internally per-epoch instead), so the
    FileReader it builds starts out with the exact same order as
    dict_lister_trains[k] itself.
    """
    root = _make_basic_ct_dir(tmp_path, num_files=20)
    module = _make_module(root, 0.0, 1.0)
    module.batches_per_rank_epoch = {"ct1": 5}
    module.setup()

    k = next(iter(module.dict_data_train))
    file_reader = module.dict_data_train[k].dataset.dataset.dataset
    assert file_reader.master_file_list == module.dict_lister_trains["ct1"]
