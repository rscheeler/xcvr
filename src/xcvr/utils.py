import numpy as np
import skrf
import xarray as xr
from pint import Quantity
from xarray import DataArray

from . import ureg


def kw2da(**kwargs):
    """
    Convert kwargs of one-dimensional data into an xarray DataArray object.

    Returns
    -------
    das : dict
        Dictionary of DataArray
    """
    # Initialize output dict
    out = dict()
    # Iterate over kwargs
    for k, v in kwargs.items():
        # Dimensional coordinate data will have quantities removed, convert to base units
        if isinstance(v, Quantity):
            v = v.to_base_units()
        out[k] = DataArray(v, dims=(k,), coords={k: v})

    return out


def snp2da(path, outport: int, inport: int) -> DataArray:
    """
    Load touchstone data (.sNp) into an xarray DataArray object. Uses scikit-rf to load and manipulate data.
    """
    # Create network object
    sfile = skrf.Network(path)

    # Get decibel data
    s = sfile.s_db[:, outport, inport] * ureg.dB
    # Get frequency data
    fs = sfile.f * ureg(sfile.frequency.unit)
    # Create DataArray
    s = DataArray(s, dims=("frequency",), coords=dict(frequency=fs))
    return s


def net2da(network: skrf.Network) -> DataArray:
    """
    Convert a skrf.Network object to a DataArray object where the data is magntiude in decibels.
    """
    # Get frequency
    fs = network.f * ureg(network.frequency.unit)
    # Create DataArray
    sda = DataArray(
        network.s_db * ureg.dB,
        dims=("frequency", "out_port", "in_port"),
        coords=dict(frequency=fs, out_port=range(2), in_port=range(2)),
    )
    return sda


def xrnan2inf(x: DataArray) -> DataArray:
    """
    Converts nans to infs. Occurs when interpolating data with infs for p1db, psat, and ip3.
    """
    if isinstance(x.data, Quantity):
        xinf = xr.where(np.isnan(x), np.inf * x.data.units, x)
    else:
        xinf = xr.where(np.isnan(x), np.inf, x)
    return xinf
