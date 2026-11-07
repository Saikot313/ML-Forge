from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from mlforge.exceptions import DeviceError
from mlforge.logging_utils import get_logger

logger = get_logger(__name__)


def resolve_device(preference: str | torch.device = "auto") -> torch.device:
    if isinstance(preference, torch.device):
        device = preference
    else:
        name = str(preference).lower()
        if name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            device = torch.device(name)
        except (RuntimeError, ValueError) as exc:
            raise DeviceError(f"Invalid device specification: '{preference}'.") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise DeviceError("CUDA device requested but CUDA is not available on this machine.")
    return device


def available_devices(requested: Sequence[str]) -> list[torch.device]:
    devices: list[torch.device] = []
    for name in requested:
        try:
            device = torch.device(name)
        except (RuntimeError, ValueError) as exc:
            raise DeviceError(f"Invalid device specification: '{name}'.") from exc
        if device.type == "cuda" and not torch.cuda.is_available():
            logger.info("Skipping device '%s': CUDA not available.", name)
            continue
        if device not in devices:
            devices.append(device)
    return devices


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def resolve_pin_memory(setting: bool | str, device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    if isinstance(setting, str):
        return setting.lower() in ("auto", "true")
    return bool(setting)


def describe_device(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {"type": device.type, "torch_version": str(torch.__version__)}
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        info["name"] = torch.cuda.get_device_name(index)
        info["cuda_version"] = torch.version.cuda
    return info