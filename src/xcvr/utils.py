import numpy as np
import skrf
import xarray as xr
from pint import Quantity
from xrench.units import ureg


def snp2da(path: str, outport: int, inport: int) -> xr.DataArray:
    """Load touchstone data (.sNp) into an xarray DataArray object.
    Uses scikit-rf to load and manipulate data.
    """
    # Create network object
    sfile = skrf.Network(path)

    # Get decibel data
    s = sfile.s_db[:, outport, inport] * ureg.dB
    # Get frequency data
    fs = sfile.f * ureg(sfile.frequency.unit)
    # Return DataArray
    return xr.DataArray(s, dims=("frequency",), coords={"frequency": fs})


def net2da(network: skrf.Network) -> xr.DataArray:
    """Convert a skrf.Network object to a DataArray object where the data is magntiude
    in decibels.
    """
    # Get frequency
    fs = network.f * ureg(network.frequency.unit)
    # Return DataArray
    return xr.DataArray(
        network.s_db * ureg.dB,
        dims=("frequency", "out_port", "in_port"),
        coords={"frequency": fs, "out_port": range(2), "in_port": range(2)},
    )


def xrnan2inf(x: xr.DataArray) -> xr.DataArray:
    """Converts nans to infs. Occurs when interpolating data with infs for p1db, psat, and ip3."""
    if isinstance(x.data, Quantity):
        units = x.data.units
        xinf = xr.where(np.isnan(x), np.inf * units, x)
        # xr.where strips pint units from the result - reattach if needed
        if not isinstance(xinf.data, Quantity):
            xinf.data = xinf.data * units
    else:
        xinf = xr.where(np.isnan(x), np.inf, x)
    return xinf


def create_pseudo_s_matrix(
    gain: xr.DataArray,
    vswr_in: xr.DataArray,
    vswr_out: xr.DataArray,
    directivity: xr.DataArray,
    delays=[0, 0, 0, 0],
) -> skrf.Network:
    """Create a pseudo s-matrix from measured network parameters. Magnitude only."""
    num_freqs = gain.size

    # 1. Magnitudes from VSWR
    s11_mag = (vswr_in - 1) / (vswr_in + 1)
    s22_mag = (vswr_out - 1) / (vswr_out + 1)

    # 2. Magnitude from Gain (accounting for mismatch losses)
    # g_linear = 10 ** (gain_db / 10)
    mismatch_loss = (1 - s11_mag**2) * (1 - s22_mag**2)

    s21_mag = np.sqrt(gain / mismatch_loss)

    # 3. Isolation magnitude
    s12_mag = np.sqrt(directivity.data.to("dimensionless")) / s21_mag

    # 4. Construct complex S-matrix (assuming 0 degree phase)
    s_matrix = np.zeros((num_freqs, 2, 2), dtype=complex)
    s_matrix[:, 0, 0] = s11_mag * np.exp(-1j * 2 * np.pi * gain.frequency * delays[0])
    s_matrix[:, 0, 1] = s12_mag * np.exp(-1j * 2 * np.pi * gain.frequency * delays[1])
    s_matrix[:, 1, 0] = s21_mag * np.exp(-1j * 2 * np.pi * gain.frequency * delays[2])
    s_matrix[:, 1, 1] = s22_mag * np.exp(-1j * 2 * np.pi * gain.frequency * delays[3])

    # Create scikit-rf Network Object
    freq = skrf.Frequency.from_f(gain.frequency, unit="hz")
    network = skrf.Network(frequency=freq, s=s_matrix, z0=50)
    return network
