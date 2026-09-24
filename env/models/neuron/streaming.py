"""Bounded-memory streaming for online NEURON trace windows."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import h5py
import numpy as np


class OnlineTraceWriter:
    """Append complete online windows to a chunked HDF5 trace.

    ``committed_samples`` and ``committed_windows`` are advanced only after all
    datasets for a window have been written and flushed.  If a job is
    interrupted during an append, readers can ignore any tail beyond the
    committed sample count.
    """

    FORMAT_VERSION = 1

    def __init__(
        self,
        path: str | Path,
        *,
        stage_names: Sequence[str],
        compression: str | None = "gzip",
        compression_level: int | None = 1,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._compression = compression
        self._compression_opts = (
            compression_level if compression is not None else None
        )
        self._file = h5py.File(self.path, "w")
        self._file.attrs["format"] = "neurostimenv_online_trace"
        self._file.attrs["format_version"] = self.FORMAT_VERSION
        self._file.attrs["committed_samples"] = 0
        self._file.attrs["committed_windows"] = 0
        string_dtype = h5py.string_dtype(encoding="utf-8")
        self._file.create_dataset(
            "stage_names",
            data=np.asarray(list(stage_names), dtype=object),
            dtype=string_dtype,
        )
        self._sample_count = 0
        self._window_count = 0
        self._closed = False
        self._file.flush()

    @property
    def committed_samples(self) -> int:
        return int(self._sample_count)

    @property
    def committed_windows(self) -> int:
        return int(self._window_count)

    def _create_time_dataset(
        self,
        name: str,
        *,
        dtype,
        chunk_samples: int,
    ) -> None:
        self._file.create_dataset(
            name,
            shape=(0,),
            maxshape=(None,),
            chunks=(chunk_samples,),
            dtype=dtype,
            compression=self._compression,
            compression_opts=self._compression_opts,
            shuffle=self._compression is not None,
        )

    def _create_channel_dataset(
        self,
        name: str,
        *,
        channels: int,
        dtype,
        chunk_samples: int,
    ) -> None:
        self._file.create_dataset(
            name,
            shape=(channels, 0),
            maxshape=(channels, None),
            chunks=(channels, chunk_samples),
            dtype=dtype,
            compression=self._compression,
            compression_opts=self._compression_opts,
            shuffle=self._compression is not None,
        )

    def _create_datasets(
        self,
        *,
        sample_count: int,
        eeg_channels: int,
        dipole_components: int,
    ) -> None:
        chunk_samples = max(1, min(int(sample_count), 65_536))
        self._create_time_dataset(
            "sample_time_ms", dtype=np.float64, chunk_samples=chunk_samples
        )
        self._create_channel_dataset(
            "eeg_v",
            channels=eeg_channels,
            dtype=np.float64,
            chunk_samples=chunk_samples,
        )
        self._create_channel_dataset(
            "dipole_nA_um",
            channels=dipole_components,
            dtype=np.float64,
            chunk_samples=chunk_samples,
        )
        self._create_time_dataset(
            "field_left_boundary_time_ms",
            dtype=np.float64,
            chunk_samples=chunk_samples,
        )
        self._create_time_dataset(
            "field_left_boundary_v_per_m",
            dtype=np.float64,
            chunk_samples=chunk_samples,
        )
        self._create_time_dataset(
            "stage_code", dtype=np.int8, chunk_samples=chunk_samples
        )

    @staticmethod
    def _channel_time_array(values, *, name: str, samples: int) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 1:
            array = array[np.newaxis, :]
        if array.ndim != 2 or array.shape[1] != samples:
            raise ValueError(
                f"{name} must have shape (channels, {samples}); got {array.shape}."
            )
        return array

    def append_window(
        self,
        *,
        sample_time_ms,
        eeg_v,
        dipole_nA_um,
        field_left_boundary_time_ms,
        field_left_boundary_v_per_m,
        stage_code: int,
    ) -> None:
        """Append and durably commit one complete simulation window."""
        if self._closed:
            raise RuntimeError("Cannot append to a closed OnlineTraceWriter.")

        sample_time = np.asarray(sample_time_ms, dtype=np.float64)
        field_time = np.asarray(field_left_boundary_time_ms, dtype=np.float64)
        field = np.asarray(field_left_boundary_v_per_m, dtype=np.float64)
        if sample_time.ndim != 1 or sample_time.size == 0:
            raise ValueError("sample_time_ms must be a non-empty one-dimensional array.")
        samples = int(sample_time.size)
        if field_time.shape != (samples,) or field.shape != (samples,):
            raise ValueError(
                "Field time/value arrays must contain one left-boundary value "
                "for every recorded sample."
            )
        eeg = self._channel_time_array(eeg_v, name="eeg_v", samples=samples)
        dipole = self._channel_time_array(
            dipole_nA_um, name="dipole_nA_um", samples=samples
        )
        for name, array in (
            ("sample_time_ms", sample_time),
            ("eeg_v", eeg),
            ("dipole_nA_um", dipole),
            ("field_left_boundary_time_ms", field_time),
            ("field_left_boundary_v_per_m", field),
        ):
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} contains non-finite values.")

        if "sample_time_ms" not in self._file:
            self._create_datasets(
                sample_count=samples,
                eeg_channels=int(eeg.shape[0]),
                dipole_components=int(dipole.shape[0]),
            )
        elif (
            self._file["eeg_v"].shape[0] != eeg.shape[0]
            or self._file["dipole_nA_um"].shape[0] != dipole.shape[0]
        ):
            raise ValueError("EEG or dipole channel count changed between windows.")

        start = self._sample_count
        stop = start + samples
        for name in (
            "sample_time_ms",
            "field_left_boundary_time_ms",
            "field_left_boundary_v_per_m",
            "stage_code",
        ):
            self._file[name].resize((stop,))
        for name in ("eeg_v", "dipole_nA_um"):
            dataset = self._file[name]
            dataset.resize((dataset.shape[0], stop))

        self._file["sample_time_ms"][start:stop] = sample_time
        self._file["eeg_v"][:, start:stop] = eeg
        self._file["dipole_nA_um"][:, start:stop] = dipole
        self._file["field_left_boundary_time_ms"][start:stop] = field_time
        self._file["field_left_boundary_v_per_m"][start:stop] = field
        self._file["stage_code"][start:stop] = int(stage_code)
        self._file.flush()

        self._sample_count = stop
        self._window_count += 1
        self._file.attrs["committed_samples"] = self._sample_count
        self._file.attrs["committed_windows"] = self._window_count
        self._file.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._file.flush()
        self._file.close()
        self._closed = True

    def __enter__(self) -> "OnlineTraceWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
