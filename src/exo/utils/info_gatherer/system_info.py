import os
import platform
import socket
import sys
from subprocess import CalledProcessError

import psutil
from anyio import run_process

from exo.shared.types.memory import Memory
from exo.shared.types.profiling import InterfaceType, NetworkInterfaceInfo


def get_os_version() -> str:
    """Return the OS version string for this node.

    On macOS this is the macOS version (e.g. ``"15.3"``).
    On other platforms it falls back to the platform name (e.g. ``"Linux"``).
    """
    if sys.platform == "darwin":
        version = platform.mac_ver()[0]
        return version if version else "Unknown"
    return platform.system() or "Unknown"


async def get_os_build_version() -> str:
    """Return the macOS build version string (e.g. ``"24D5055b"``).

    On non-macOS platforms, returns ``"Unknown"``.
    """
    if sys.platform != "darwin":
        return "Unknown"

    try:
        process = await run_process(["sw_vers", "-buildVersion"])
    except CalledProcessError:
        return "Unknown"

    return process.stdout.decode("utf-8", errors="replace").strip() or "Unknown"


def get_gpu_working_set() -> Memory | None:
    """The most memory this Mac's GPU should use: Metal's recommended working set, which is
    also the limit a runner wires its model's memory up to. None without Metal."""
    if sys.platform != "darwin":
        return None
    try:
        import mlx.core as mx
    except ImportError:
        return None
    if not mx.metal.is_available():
        return None
    return Memory.from_bytes(int(mx.device_info()["max_recommended_working_set_size"]))


_nvml_ready: bool | None = None


def _cuda_device_index() -> int:
    """The NVML index of the GPU MLX's CUDA backend runs on (its default device, the first
    one CUDA_VISIBLE_DEVICES exposes). NVML ignores CUDA_VISIBLE_DEVICES, so map it here."""
    visible = os.getenv("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    return int(visible) if visible.isdigit() else 0


def get_cuda_vram_available() -> Memory | None:
    """Free VRAM a model can still use on this node's CUDA GPU, less a reserve for the CUDA
    context, activations and KV cache (EXO_CUDA_VRAM_RESERVE_MB, default 1024).

    Like `ram_available`, it is read every poll, so the memory loaded models hold is already
    gone from it. None without an NVIDIA GPU (or without nvidia-ml-py)."""
    global _nvml_ready
    if sys.platform != "linux" or _nvml_ready is False:
        return None
    try:
        import pynvml as nvml  # pyright: ignore[reportMissingModuleSource]
    except ImportError:
        _nvml_ready = False
        return None
    try:
        if _nvml_ready is None:
            nvml.nvmlInit()
            _nvml_ready = nvml.nvmlDeviceGetCount() > 0
            if not _nvml_ready:
                return None
        handle = nvml.nvmlDeviceGetHandleByIndex(_cuda_device_index())
        free = int(nvml.nvmlDeviceGetMemoryInfo(handle).free)
    except Exception:
        _nvml_ready = False
        return None
    reserve = Memory.from_mb(int(os.getenv("EXO_CUDA_VRAM_RESERVE_MB", "1024"))).in_bytes
    return Memory.from_bytes(max(0, free - reserve))


async def get_friendly_name() -> str:
    """
    Asynchronously gets the 'Computer Name' (friendly name) of a Mac.
    e.g., "John's MacBook Pro"
    Returns the name as a string, or None if an error occurs or not on macOS.
    """
    hostname = socket.gethostname()

    if sys.platform != "darwin":
        return hostname

    try:
        process = await run_process(["scutil", "--get", "ComputerName"])
    except CalledProcessError:
        return hostname

    return process.stdout.decode("utf-8", errors="replace").strip() or hostname


async def _get_interface_types_from_networksetup() -> dict[str, InterfaceType]:
    """Parse networksetup -listallhardwareports to get interface types."""
    if sys.platform != "darwin":
        return {}

    try:
        result = await run_process(["networksetup", "-listallhardwareports"])
    except CalledProcessError:
        return {}

    types: dict[str, InterfaceType] = {}
    current_type: InterfaceType = "unknown"

    for line in result.stdout.decode().splitlines():
        if line.startswith("Hardware Port:"):
            port_name = line.split(":", 1)[1].strip()
            if "Wi-Fi" in port_name:
                current_type = "wifi"
            elif "Ethernet" in port_name or "LAN" in port_name:
                current_type = "ethernet"
            elif port_name.startswith("Thunderbolt"):
                current_type = "thunderbolt"
            else:
                current_type = "unknown"
        elif line.startswith("Device:"):
            device = line.split(":", 1)[1].strip()
            # enX is ethernet adapters or thunderbolt - these must be deprioritised
            if device.startswith("en") and device not in ["en0", "en1"]:
                current_type = "maybe_ethernet"
            types[device] = current_type

    return types


async def get_network_interfaces() -> list[NetworkInterfaceInfo]:
    """
    Retrieves detailed network interface information on macOS.
    Parses output from 'networksetup -listallhardwareports' and 'ifconfig'
    to determine interface names, IP addresses, and types (ethernet, wifi, vpn, other).
    Returns a list of NetworkInterfaceInfo objects.
    """
    interfaces_info: list[NetworkInterfaceInfo] = []
    interface_types = await _get_interface_types_from_networksetup()

    for iface, services in psutil.net_if_addrs().items():
        for service in services:
            match service.family:
                case socket.AF_INET | socket.AF_INET6:
                    interfaces_info.append(
                        NetworkInterfaceInfo(
                            name=iface,
                            ip_address=service.address,
                            interface_type=interface_types.get(iface, "unknown"),
                        )
                    )
                case _:
                    pass

    return interfaces_info


async def get_model_and_chip() -> tuple[str, str]:
    """Get Mac system information using system_profiler."""
    model = "Unknown Model"
    chip = "Unknown Chip"

    # TODO: better non mac support
    if sys.platform != "darwin":
        return (model, chip)

    try:
        process = await run_process(
            [
                "system_profiler",
                "SPHardwareDataType",
            ]
        )
    except CalledProcessError:
        return (model, chip)

    # less interested in errors here because this value should be hard coded
    output = process.stdout.decode().strip()

    model_line = next(
        (line for line in output.split("\n") if "Model Name" in line), None
    )
    model = model_line.split(": ")[1] if model_line else "Unknown Model"

    chip_line = next((line for line in output.split("\n") if "Chip" in line), None)
    chip = chip_line.split(": ")[1] if chip_line else "Unknown Chip"

    return (model, chip)
