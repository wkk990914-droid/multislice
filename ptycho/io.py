"""Data loading, diffraction calibration and result persistence."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from quantem.core.datastructures import Dataset4dstem
from quantem.core.utils.diffractive_imaging_utils import fit_probe_circle

from .config import DataConfig, ExperimentConfig

_DTYPE_LOOKUP = {
    "float32": np.float32,
    "float64": np.float64,
    "int32": np.int32,
    "uint16": np.uint16,
}


def resolve_dtype(name: str) -> np.dtype:
    try:
        return np.dtype(_DTYPE_LOOKUP.get(name, name))
    except TypeError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported dtype {name!r}") from exc


def load_raw(
    file_path: str | Path,
    shape: tuple[int, int, int],
    dtype: np.dtype = np.float32,
    offset: int = 0,
    gap: int = 1024,
) -> np.ndarray:
    """Load an EMPAD-style ``.raw`` file with inter-frame gaps.

    Adapted from ``ptyrad``: reading with a custom structured dtype skips the
    gap bytes in one pass, roughly 2x faster than read + reshape.

    Parameters
    ----------
    file_path
        Path to the raw file.
    shape
        ``(N, height, width)`` with ``N`` the number of frames.
    dtype
        Per-element numpy dtype.
    offset
        Bytes to skip at the beginning of the file.
    gap
        Number of padding bytes between consecutive frames.

    Returns
    -------
    np.ndarray
        Array of shape ``(N, height, width)``.
    """
    file_path = Path(file_path).expanduser()
    n_frames, height, width = shape

    expected_size = offset + n_frames * (height * width * dtype.itemsize + gap)
    actual_size = os.path.getsize(file_path)
    if actual_size != expected_size:
        raise ValueError(
            "Mismatch in expected "
            f"({expected_size} bytes = offset + N * (height * width * "
            f"{dtype.itemsize} + gap)) vs. actual ({actual_size} bytes) file "
            "size! Check your loading configurations!"
        )

    custom_dtype = np.dtype(
        [
            ("data", dtype, (height, width)),
            ("gap", np.uint8, gap),
        ]
    )

    with open(file_path, "rb") as handle:
        handle.seek(offset)
        raw = np.fromfile(handle, dtype=custom_dtype, count=n_frames)

    data = raw["data"]
    print("Success! Loaded .raw file path =", file_path)
    print("Imported .raw data shape =", data.shape)
    print("Imported .raw data type =", data.dtype)
    return data


_H5_SUFFIXES = {".h5", ".hdf5", ".hdf", ".nexus"}


def _find_4d_dataset(h5_group: Any, preferred: str | None = None) -> Any:
    """Return the first 4D dataset under ``h5_group`` (or ``preferred`` path)."""
    import h5py

    if preferred:
        obj = h5_group[preferred]
        if isinstance(obj, h5py.Dataset) and obj.ndim == 4:
            return obj
        raise ValueError(f"Dataset {preferred!r} is not 4D (ndim={getattr(obj, 'ndim', '?')})")

    found: list[Any] = []
    h5_group.visit(lambda name, o: found.append(o) if isinstance(o, h5py.Dataset) and o.ndim == 4 else None)
    if not found:
        raise ValueError(f"No 4D dataset found in {h5_group.name!r}")
    if len(found) > 1:
        print(f"Multiple 4D datasets found: {[d.name for d in found]}; using {found[0].name}")
    return found[0]


def _axis_order_from_h5(dataset: Any) -> tuple[int, ...]:
    """Infer (R0,R1,R2,R3)->source-axis permutation from HDF5 axis labels."""
    labels = []
    for i in range(4):
        meta = dataset.attrs.get(f"Axis{i}_quantity", "")
        if isinstance(meta, bytes):
            meta = meta.decode()
        labels.append(str(meta).lower())

    def pick(*keywords: str) -> int | None:
        for k, lab in enumerate(labels):
            if any(w in lab for w in keywords):
                return k
        return None

    rx = pick("x_", "position_x", "scan_x")
    ry = pick("y_", "position_y", "scan_y")
    qx = pick("q_x", "k_x", "x_pixel")
    qy = pick("q_y", "k_y", "y_pixel")
    if None in (rx, ry, qx, qy):
        return (0, 1, 2, 3)
    order = (ry, rx, qy, qx)
    if len(set(order)) != 4:
        return (0, 1, 2, 3)
    return order


def _apply_crops(array: np.ndarray, data_cfg: DataConfig) -> np.ndarray:
    """Slice a ``(ny, nx, qy, qx)`` array down to the configured sub-region."""
    if data_cfg.scan_crop:
        y0, y1, x0, x1 = data_cfg.scan_crop
        array = array[y0:y1, x0:x1, :, :]
    if data_cfg.dp_crop:
        qy0, qy1, qx0, qx1 = data_cfg.dp_crop
        array = array[:, :, qy0:qy1, qx0:qx1]
    return np.ascontiguousarray(array)


def _apply_scan_axis_convention(array: np.ndarray, data_cfg: DataConfig) -> np.ndarray:
    if data_cfg.transpose_scan:
        array = np.transpose(array, (1, 0, 2, 3))
    if data_cfg.flip_scan_y:
        array = array[::-1, :, :, :]
    if data_cfg.flip_scan_x:
        array = array[:, ::-1, :, :]
    return np.ascontiguousarray(array)


def _apply_dp_axis_convention(array: np.ndarray, data_cfg: DataConfig) -> np.ndarray:
    if data_cfg.transpose_dp:
        array = np.transpose(array, (0, 1, 3, 2))
    if data_cfg.flip_dp_y:
        array = array[:, :, ::-1, :]
    if data_cfg.flip_dp_x:
        array = array[:, :, :, ::-1]
    return np.ascontiguousarray(array)


def load_hdf5_scan(data_cfg: DataConfig) -> np.ndarray:
    """Load a (PYREX-style) HDF5 4D-STEM file into ``(ny, nx, qy, qx)``.

    When the axis order is already ``(Ry, Rx, Qy, Qx)`` and crops are configured
    the sub-region is read straight from disk as a hyperslab, which avoids
    materialising the full 4D volume in memory.
    """
    import h5py

    path = data_cfg.resolved_path()
    with h5py.File(path, "r") as f:
        root = f[data_cfg.group] if data_cfg.group else f
        ds = _find_4d_dataset(root, data_cfg.dataset)
        order = _axis_order_from_h5(ds) if data_cfg.autodetect_axes else (0, 1, 2, 3)
        print(f"HDF5 dataset {ds.name} shape={ds.shape} dtype={ds.dtype} axis order={order}")

        if order == (0, 1, 2, 3) and (data_cfg.scan_crop or data_cfg.dp_crop):
            y0, y1, x0, x1 = data_cfg.scan_crop or (0, ds.shape[0], 0, ds.shape[1])
            qy0, qy1, qx0, qx1 = data_cfg.dp_crop or (0, ds.shape[2], 0, ds.shape[3])
            array = ds[y0:y1, x0:x1, qy0:qy1, qx0:qx1]
            print("Loaded 4D-STEM hyperslab shape (Ry,Rx,Qy,Qx) =", array.shape)
            array = np.ascontiguousarray(array, dtype=np.float32)
            array = _apply_scan_axis_convention(array, data_cfg)
            array = _apply_dp_axis_convention(array, data_cfg)
            return array

        array = ds[...]

    array = np.transpose(np.asarray(array), order)
    print("Loaded 4D-STEM shape (Ry,Rx,Qy,Qx) =", array.shape, array.dtype)
    array = _apply_crops(array.astype(np.float32, copy=False), data_cfg)
    array = _apply_scan_axis_convention(array, data_cfg)
    print("Working 4D-STEM shape (Ry,Rx,Qy,Qx) =", array.shape)
    return _apply_dp_axis_convention(array, data_cfg)


def load_scan(data_cfg: DataConfig) -> np.ndarray:
    """Load the raw/HDF5 file and fold it into a ``(ny, nx, qy, qx)`` array."""
    path = data_cfg.resolved_path()
    if path.suffix.lower() in _H5_SUFFIXES:
        return load_hdf5_scan(data_cfg)

    ny, nx = data_cfg.scan_shape
    qy, qx = data_cfg.dp_shape
    dtype = resolve_dtype(data_cfg.dtype)

    array = load_raw(
        path,
        shape=(data_cfg.num_frames, qy, qx),
        dtype=dtype,
        offset=data_cfg.offset_bytes,
        gap=data_cfg.gap_bytes,
    )
    array = array.reshape((ny, nx, qy, qx))
    array = _apply_crops(array, data_cfg)
    array = _apply_scan_axis_convention(array, data_cfg)
    return _apply_dp_axis_convention(array, data_cfg)


def build_dataset4dstem(
    array: np.ndarray, data_cfg: DataConfig, exp_cfg: ExperimentConfig
) -> Dataset4dstem:
    """Wrap the array in a :class:`Dataset4dstem` using the real-space sampling."""
    step = exp_cfg.scan_step_size
    return Dataset4dstem.from_array(
        array,
        name=data_cfg.name,
        sampling=(step, step, 1, 1),
        units=("A", "A", "pixels", "pixels"),
    )


def calibrate_diffraction_sampling(
    dset: Dataset4dstem,
    exp_cfg: ExperimentConfig,
    data_cfg: DataConfig | None = None,
) -> tuple[float, float, float]:
    """Calibrate the diffraction axes to mrad.

    With an explicit ``data_cfg.dp_sampling_mrad`` the geometric centre is used
    (simulation data); otherwise the probe disc is fitted on the mean DP.

    Returns ``(qy0, qx0, radius)`` in pixels.
    """
    if data_cfg is not None and data_cfg.dp_sampling_mrad:
        sampling = data_cfg.dp_sampling_mrad
        qy, qx = dset.array.shape[-2:]
        probe_qy0, probe_qx0 = qy / 2.0, qx / 2.0
        probe_radius = exp_cfg.probe_semiangle / sampling
        dset.sampling[2] = sampling
        dset.sampling[3] = sampling
        dset.units[2:] = ["mrad", "mrad"]
        dset.get_dp_mean()
        return probe_qy0, probe_qx0, probe_radius

    probe_qy0, probe_qx0, probe_radius = fit_probe_circle(dset.dp_mean.array, show=True)
    reciprocal_sampling = exp_cfg.probe_semiangle / probe_radius
    dset.sampling[2] = reciprocal_sampling
    dset.sampling[3] = reciprocal_sampling
    dset.units[2:] = ["mrad", "mrad"]
    dset.get_dp_mean()
    return probe_qy0, probe_qx0, probe_radius


def to_numpy(array: Any) -> np.ndarray:
    """Convert torch tensors / quantem wrappers to a plain numpy array."""
    if array is None:
        raise ValueError("Cannot convert None to numpy array")
    if isinstance(array, np.ndarray):
        return array
    if hasattr(array, "numpy"):  # torch.Tensor on cpu
        return array.detach().cpu().numpy()
    if hasattr(array, "array"):  # quantem datastructures
        return to_numpy(array.array)
    return np.asarray(array)


def save_array(array: Any, path: str | Path) -> Path:
    """Save an array (or tensor) to ``.npy``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, to_numpy(array))
    return path
