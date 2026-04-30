"""Conversions module."""

from typing import Union

from pint import Quantity
from xarray import DataArray


def nf2temp(nf: Quantity | DataArray, t0: Quantity) -> Quantity | DataArray:
    """
    Converts noise figure to noise temperature.

    Parameters
    ----------
    nf : Quantity, DataArray
        Noise figure
    t0 : Quantity, DataArray
        Reference temperature

    Returns:
    -------
    te : Quantity, DataArray
        Equivalent input noise temperature.
    """
    return t0 * (nf - 1)


def temp2nf(temp: Quantity | DataArray, t0: Quantity) -> Quantity | DataArray:
    """
    Converts equivalent input noise temperature to noise figure.

    Parameters
    ----------
    temp : Quantity, DataArray
        Equivalent input noise temperature
    t0 : Quantity, DataArray
        Reference temperature.

    Returns:
    -------
    nf : Quantity, DataArray
        Noise figure
    """
    return temp / t0 + 1
