import os
from typing import Literal

from torch.utils.data import Dataset

from flow_control.utils.pipeline import DataSink

from .directory import (
    PickleDirectoryDataset,
    PickleDirectoryDataSink,
    RawDirectoryDataset,
    RawDirectoryDataSink,
)


class BinsDirectoryDataset(Dataset):
    bin_lengths: list[int]

    def __init__(self, path: str, base_type: Literal["raw", "pickle"], **kwargs):
        bin_dirs = []
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_dir():
                    bin_dirs.append(entry.path)
        bin_dirs = sorted(bin_dirs)
        self.datasets = [
            RawDirectoryDataset(d, **kwargs)
            if base_type == "raw"
            else PickleDirectoryDataset(d, **kwargs)
            for d in bin_dirs
        ]
        self.bin_lengths = [len(ds) for ds in self.datasets]
        self.cum_lengths = []
        cum_length = 0
        for length in self.bin_lengths:
            self.cum_lengths.append(cum_length)
            cum_length += length

    def __len__(self):
        return sum(self.bin_lengths)

    def __getitem__(self, index: int):
        for i, cum_length in enumerate(self.cum_lengths):
            if index < cum_length + self.bin_lengths[i]:
                return self.datasets[i][index - cum_length]
        raise IndexError("Index out of range")


class BinsDirectoryDataSink(DataSink):
    def __init__(
        self,
        worker_id,
        path: str,
        bin_spec: int | list[int],
        base_type: Literal["raw", "pickle"],
        **kwargs,
    ):
        self.worker_id = worker_id
        self.path = path
        self.bin_spec = bin_spec
        self.base_type = base_type
        self.kwargs = kwargs
        self.data_sinks = {}

        if self.bin_spec is list and self.bin_spec[0] != 0:
            self.bin_spec = [0] + self.bin_spec

        os.makedirs(self.path, exist_ok=True)

    def _get_data_sink(self, latent_length: int | None = None):
        if latent_length is None:
            bin = "unknown"
        elif isinstance(self.bin_spec, int):
            bin = "bin_" + str(latent_length // self.bin_spec * self.bin_spec)
        else:
            bin = "bin_" + str(max(b for b in self.bin_spec if b <= latent_length))
        if bin not in self.data_sinks:
            bin_dir = os.path.join(self.path, bin)
            self.data_sinks[bin] = (
                RawDirectoryDataSink(self.worker_id, bin_dir, **self.kwargs)
                if self.base_type == "raw"
                else PickleDirectoryDataSink(self.worker_id, bin_dir, **self.kwargs)
            )
        return self.data_sinks[bin]

    def write(self, data: dict):
        latent_length = data.get("latent_length")
        data_sink = self._get_data_sink(latent_length)
        data_sink.write(data)
        return True