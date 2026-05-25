from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from xarray import DataArray

from xcvr.xcvr import System


def plot_cascade(
    rfsys: System,
    prop: str = "gain",
    xtick_params: dict = dict(rotation=30),
    **kwargs: Any,
) -> plt.Axes:
    """Plot cascaded parameter."""
    # Get cascaded property
    cp = rfsys.__getattribute__(f"cascaded_{prop}").sel(**kwargs)

    # Create figure
    fig, ax = plt.subplots()
    # Plot data
    cp.plot(ax=ax)
    # Format
    ax.xaxis.set_tick_params(**xtick_params)
    fig.set_figwidth(fig.get_figwidth() * 1.5)
    # Set Title
    ax.set_title(prop)
    return ax


def plot_signal_compression(
    rfsys: System,
    input_pwr: DataArray,
    xtick_params: dict = {"rotation": 30},
    **kwargs: Any,
) -> plt.Axes:
    """Plot signal level and compression through system."""
    # Get signal level and cascaded compression values
    siglev = rfsys.get_signal_level(input_pwr.sel(**kwargs)).sel(**kwargs)
    p1db = rfsys.cascaded_p1db.sel(**kwargs)
    psat = rfsys.cascaded_psat.sel(**kwargs)
    psatdev = rfsys.get_device_attr("psat").sel(**kwargs)

    # Ensure units are in dBm
    siglev.data = siglev.data.to("dBm")
    p1db.data = p1db.data.to("dBm")

    # Plot the signal level
    fig, ax = plt.subplots()
    y = siglev.plot.line(ax=ax, x="device", label="Signal Level")
    y1 = p1db.plot.line(ax=ax, x="device", color="C1", ls="--", label="P1dB")
    ysat = psat.plot.line(ax=ax, x="device", color="C3", ls="--", label="Psat")
    ysatdev = psatdev.plot.line(
        ax=ax,
        x="device",
        color="C3",
        ls="",
        marker="o",
        label="Device Psat",
    )
    # Add label
    ax.set_ylabel("Power [dBm]")
    # Format
    ax.xaxis.set_tick_params(**xtick_params)
    fig.set_figwidth(fig.get_figwidth() * 1.5)
    props = dict(boxstyle="round", facecolor="white", alpha=0.8)
    fontsize = 10
    ax.text(
        0.1,
        0.6,
        f"Input Power: {input_pwr.sel(**kwargs).item().to('dBm').magnitude:.2f} dBm",
        transform=ax.transAxes,
        fontsize=fontsize,
        bbox=props,
    )

    # Show headroom
    y = np.nan_to_num(y[0].get_ydata(), posinf=np.nan)
    y1 = np.nan_to_num(y1[0].get_ydata(), posinf=np.nan)
    ax.fill_between(siglev.device, y, y1, where=(y > y1), color="C3", alpha=0.3, interpolate=True)
    ax.fill_between(siglev.device, y, y1, where=(y < y1), color="C0", alpha=0.3, interpolate=True)
    ax.legend()
    ax.set_title("Signal Level and Compression")
    return ax
