"""xcvr: Python library for RF transceiver cascaded analysis."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("xcvr")
except PackageNotFoundError:
    # Package is not installed
    __version__ = "0.0.0-dev"

from xcvr.devices import Cable, Constant
from xcvr.plotting import plot_cascade, plot_signal_compression
from xcvr.symbols import Symbol
from xcvr.xcvr import Device, System

__all__ = [
    "Cable",
    "Constant",
    "Device",
    "Symbol",
    "System",
    "plot_cascade",
    "plot_signal_compression",
]
