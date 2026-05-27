"""Helpers for loading and synchronizing Neuracore datasets."""

from typing import Any

import neuracore as nc
from neuracore.core.utils.robot_data_spec_utils import (
    merge_cross_embodiment_description,
)
from neuracore_types import CrossEmbodimentDescription, DataType


def load_and_sync_dataset(
    dataset_name: str,
    frequency: int,
    input_modalities: list[DataType] | None = None,
    output_modalities: list[DataType] | None = None,
    prefetch_videos: bool = False,
) -> Any:
    """Loads a Neuracore dataset and synchronizes the specified modalities across embodiments."""
    print(f"\n🔍 Getting dataset '{dataset_name}' from Neuracore...")
    dataset = nc.get_dataset(dataset_name)

    print("🔁 Building cross_embodiment_union for synchronization...")
    input_cross_embodiment: CrossEmbodimentDescription = {}
    output_cross_embodiment: CrossEmbodimentDescription = {}

    for robot_id in dataset.robot_ids:
        full = dataset.get_full_embodiment_description(robot_id)
        if input_modalities:
            input_cross_embodiment[robot_id] = {
                dt: full[dt] for dt in input_modalities if dt in full
            }
        if output_modalities:
            output_cross_embodiment[robot_id] = {
                dt: full[dt] for dt in output_modalities if dt in full
            }

    cross_embodiment_union = merge_cross_embodiment_description(
        input_cross_embodiment, output_cross_embodiment
    )

    print("🔁 Synchronizing dataset...")

    # Dynamically build arguments to avoid passing 0 workers to the ThreadPoolExecutor
    sync_kwargs = {
        "frequency": frequency,
        "cross_embodiment_union": cross_embodiment_union,
        "prefetch_videos": prefetch_videos,
    }

    if prefetch_videos:
        sync_kwargs["max_prefetch_workers"] = 2

    synced_dataset = dataset.synchronize(**sync_kwargs)

    print(f"  ✓ Dataset synchronized: {len(synced_dataset)} episodes")
    return synced_dataset
