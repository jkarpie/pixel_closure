"""Small tests for the full-scale per-dataset kernel process farm."""

from dataclasses import dataclass

import kernel_farm


@dataclass
class FakeDataset:
    name: str


def fake_assembler(dataset, fields):
    import os

    return dataset.name, fields["marker"], os.environ["PIXEL_KERNEL_WORKERS"], os.environ["PIXEL_ASSEMBLY_WORKERS"]


def test_process_farm_preserves_order_and_disables_nested_pools(monkeypatch):
    monkeypatch.setenv("PIXEL_DATASET_WORKERS", "2")
    monkeypatch.setenv("PIXEL_KERNEL_WORKERS", "9")
    monkeypatch.setenv("PIXEL_ASSEMBLY_WORKERS", "9")
    monkeypatch.setattr(kernel_farm, "_available_bytes", lambda: 100 * 1024 ** 3)
    datasets = [FakeDataset(name) for name in ("first", "second", "third")]

    result = kernel_farm.assemble_datasets(datasets, {"marker": 42}, fake_assembler)

    assert result == [(name, (name, 42, "1", "1")) for name in ("first", "second", "third")]


def test_worker_count_respects_memory_budget(monkeypatch):
    monkeypatch.setenv("PIXEL_DATASET_WORKERS", "32")
    monkeypatch.setattr(kernel_farm, "_available_bytes", lambda: 24 * 1024 ** 3)
    assert kernel_farm.worker_count(46) == 4


def test_cached_small_closure_kernels_work_in_spawned_processes(monkeypatch):
    """Exercise actual PIXEL Dataset pickling without full data generation."""
    from closure_JAM_truth_small import config, datasets, generate

    fields = config.make_fields()
    lattice = datasets.build_datasets(
        "2", fields, include_exp=False, with_exp_nuisances=False
    )[0]
    experiment = datasets.build_datasets(
        "2", fields, include_lattice=False, with_exp_nuisances=False
    )[0]
    monkeypatch.setenv("PIXEL_DATASET_WORKERS", "2")
    monkeypatch.setattr(kernel_farm, "_available_bytes", lambda: 100 * 1024 ** 3)

    operators = kernel_farm.assemble_datasets(
        [lattice, experiment], fields, generate.assemble_operator
    )

    assert [name for name, _ in operators] == [lattice.name, experiment.name]
    assert all(operator for _, operator in operators)
