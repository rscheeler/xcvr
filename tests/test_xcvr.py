import numpy as np
import pytest
import xarray as xr
from xrench.units import ureg

from xcvr.devices import Cable, Constant
from xcvr.xcvr import System


@pytest.fixture
def frequency():
    """Standard 3-point frequency array for testing."""
    f_vals = [1e9, 1.5e9, 2e9]
    return xr.DataArray(
        f_vals * ureg.Hz,
        dims=("frequency",),
        coords={"frequency": f_vals},
    )


def test_constant_device_initialization(frequency):
    """Test that a Constant device initializes with correct units and shape."""
    gain_val = 20 * ureg.dB
    nf_val = 3 * ureg.dB

    amp = Constant(
        name="LNA",
        manufacturer="Generic",
        pn="AMP1",
        frequency=frequency,
        gain=gain_val,
        nf=nf_val,
    )

    # Check dimensions
    assert "frequency" in amp.gain.dims
    assert amp.gain.size == 3

    # Check value (magnitude and unit)
    assert np.allclose(amp.gain.values, 20.0)
    assert str(amp.gain.data.units) == "decibel"


def test_system_cascade_math(frequency):
    """Verify cascaded Gain and Noise Figure math."""
    # Stage 1: 10dB Gain, 2dB NF
    amp1 = Constant("A1", "M", "P", frequency, gain=10 * ureg.dB, nf=2 * ureg.dB)
    # Stage 2: 10dB Gain, 5dB NF
    amp2 = Constant("A2", "M", "P", frequency, gain=10 * ureg.dB, nf=5 * ureg.dB)

    sys = System("TestChain", "xcvr inc.", "001", [amp1, amp2])

    # Cascaded Gain = G1 + G2 = 10 + 10 = 20dB
    assert np.allclose(sys.gain.values, 20.0)

    # Friis Equation for NF: F_total = F1 + (F2 - 1) / G1
    # F1_lin = 10^(2/10) = 1.5849
    # F2_lin = 10^(5/10) = 3.1622
    # G1_lin = 10^(10/10) = 10.0
    # F_sys = 1.5849 + (3.1622 - 1) / 10 = 1.5849 + 0.21622 = 1.8011
    # NF_sys = 10 * log10(1.8011) = 2.555 dB
    assert np.allclose(sys.nf.values, 2.555, atol=1e-3)


def test_interpolation_fallback_fix(frequency):
    """
    Test that the system doesn't crash on small frequency arrays.
    Regression test for the ValueError in scipy.interpolate.
    """
    # Create a 2-point array (used to cause 'expected 1 derivative' error)
    small_f = xr.DataArray(
        [1e9, 2e9] * ureg.Hz,
        dims=("frequency",),
        coords={"frequency": [1e9, 2e9]},
    )

    # This should not raise ValueError
    try:
        amp = Constant("A1", "M", "P", small_f, gain=10 * ureg.dB)
    except ValueError as e:
        pytest.fail(f"Interpolation failed on small array: {e}")


def test_cable_device(frequency):
    """Test Cable device loss calculation."""
    length = 10 * ureg.m
    # Highly lossy tan_delta for clear measurement
    cable = Cable(
        "TestCable",
        "Generic",
        "C1",
        frequency,
        length=length,
        coaxial_kwargs=dict(tan_delta=0.5),
    )

    # Gain should be negative (loss)
    assert np.all(cable.gain.values < 0)
    # Check that loss increases with frequency
    assert cable.gain.values[0] > cable.gain.values[-1]


def test_system_with_subsystem(frequency):
    """Test nesting systems within systems."""
    amp = Constant("Amp", "M", "P", frequency, gain=10 * ureg.dB)
    sub_sys = System("Sub", "xcvr inc.", "001", [amp, amp])  # 20dB gain

    top_sys = System("Top", "xcvr inc.", "001", [sub_sys, amp])  # 20 + 10 = 30dB gain
    assert np.allclose(top_sys.gain.values, 30.0)
