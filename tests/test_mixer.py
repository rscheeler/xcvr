"""
Tests for mixer and FrequencyPlan frequency translation in xcvr.

Assumes:
    - xcvr.xcvr contains Device, System, MixerMixin, FrequencyPlan
    - xcvr.devices contains Mixer (and optionally MixerConstant)
    - xrench.units provides ureg
"""

import numpy as np
import pytest
import skrf
from xrench.units import ureg

from xcvr.devices import Mixer, MixerConstant
from xcvr.xcvr import Device, FrequencyPlan, MixerMixin, System

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_network(freqs_hz: np.ndarray, s21_db: float = 10.0, s11_db: float = -20.0) -> skrf.Network:
    """Create a simple 2-port network with flat S-parameters."""
    npts = len(freqs_hz)
    s = np.zeros((npts, 2, 2), dtype=complex)
    s21_lin = 10 ** (s21_db / 20)
    s11_lin = 10 ** (s11_db / 20)
    s[:, 1, 0] = s21_lin  # S21
    s[:, 0, 1] = s21_lin  # S12
    s[:, 0, 0] = s11_lin  # S11
    s[:, 1, 1] = s11_lin  # S22
    freq = skrf.Frequency.from_f(freqs_hz, unit="hz")
    return skrf.Network(frequency=freq, s=s)


def make_device(
    freqs_hz: np.ndarray,
    name: str = "AMP",
    s21_db: float = 10.0,
    nf_db: float = 3.0,
    p1db_dbm: float = 20.0,
) -> Device:
    """Create a Device with flat gain, NF, and P1dB."""
    net = make_network(freqs_hz, s21_db=s21_db)
    nf = skrf.network.Network  # not used directly
    nf_da = None  # let Device compute from network
    return Device(
        name=name,
        manufacturer="Test",
        pn="TEST-001",
        network=net,
        nf=ureg.Quantity(nf_db, "dB"),
        p1db=ureg.Quantity(p1db_dbm, "dBm"),
    )


def make_mixer(
    freqs_hz: np.ndarray,
    lo_freq: float,
    sideband: str = "high",
    name: str = "MXR",
    conversion_loss_db: float = 7.0,
) -> Mixer:
    """Create a Mixer with flat conversion loss."""
    net = make_network(freqs_hz, s21_db=-conversion_loss_db)
    return Mixer(
        name=name,
        manufacturer="Test",
        pn="MXR-001",
        network=net,
        lo_freq=lo_freq,
        sideband=sideband,
        nf=ureg.Quantity(conversion_loss_db, "dB"),
    )


# ---------------------------------------------------------------------------
# FrequencyPlan unit tests
# ---------------------------------------------------------------------------


class TestFrequencyPlan:
    def test_passthrough_rf_equals_carrier(self):
        freqs = np.linspace(1e9, 2e9, 101)
        fp = FrequencyPlan.passthrough(freqs)
        assert np.array_equal(fp.rf, fp.carrier)

    def test_is_translated_false_for_passthrough(self):
        freqs = np.linspace(1e9, 2e9, 101)
        fp = FrequencyPlan.passthrough(freqs)
        assert not fp.is_translated

    def test_translate_high_sideband(self):
        freqs = np.linspace(3e9, 4e9, 101)
        lo = 3.5e9
        fp = FrequencyPlan.passthrough(freqs)
        translated = fp.translate(lo, sideband="high")
        expected = freqs + lo
        np.testing.assert_allclose(translated.carrier, expected)

    def test_translate_low_sideband(self):
        freqs = np.linspace(3e9, 4e9, 101)
        lo = 3.5e9
        fp = FrequencyPlan.passthrough(freqs)
        translated = fp.translate(lo, sideband="low")
        expected = np.abs(freqs - lo)
        np.testing.assert_allclose(translated.carrier, expected)

    def test_translate_preserves_rf(self):
        freqs = np.linspace(3e9, 4e9, 101)
        fp = FrequencyPlan.passthrough(freqs)
        translated = fp.translate(3.5e9, sideband="high")
        np.testing.assert_array_equal(translated.rf, freqs)

    def test_is_translated_true_after_translate(self):
        freqs = np.linspace(3e9, 4e9, 101)
        fp = FrequencyPlan.passthrough(freqs).translate(3.5e9, sideband="high")
        assert fp.is_translated

    def test_label_attaches_carrier_freq_coord(self):
        import xarray as xr

        freqs = np.linspace(3e9, 4e9, 11)
        lo = 3.5e9
        fp = FrequencyPlan.passthrough(freqs).translate(lo, sideband="high")
        da = xr.DataArray(np.ones(11), dims=("frequency",), coords={"frequency": freqs})
        labeled = fp.label(da)
        assert "carrier_freq" in labeled.coords
        np.testing.assert_allclose(labeled.coords["carrier_freq"].values, freqs + lo)

    def test_label_rf_axis_unchanged(self):
        import xarray as xr

        freqs = np.linspace(3e9, 4e9, 11)
        fp = FrequencyPlan.passthrough(freqs).translate(3.5e9, sideband="high")
        da = xr.DataArray(np.ones(11), dims=("frequency",), coords={"frequency": freqs})
        labeled = fp.label(da)
        np.testing.assert_array_equal(labeled.coords["frequency"].values, freqs)


# ---------------------------------------------------------------------------
# MixerMixin unit tests
# ---------------------------------------------------------------------------


class TestMixerMixin:
    def test_mixer_is_instance_of_mixin(self):
        freqs = np.linspace(3e9, 4e9, 51)
        mixer = make_mixer(freqs, lo_freq=3.5e9)
        assert isinstance(mixer, MixerMixin)

    def test_mixer_stores_lo_freq(self):
        freqs = np.linspace(3e9, 4e9, 51)
        lo = 3.5e9
        mixer = make_mixer(freqs, lo_freq=lo)
        assert mixer.lo_freq == lo

    def test_mixer_stores_sideband(self):
        freqs = np.linspace(3e9, 4e9, 51)
        mixer = make_mixer(freqs, lo_freq=3.5e9, sideband="low")
        assert mixer.sideband == "low"

    def test_mixer_default_sideband(self):
        freqs = np.linspace(3e9, 4e9, 51)
        mixer = make_mixer(freqs, lo_freq=3.5e9)
        assert mixer.sideband in ("high", "low")  # just check it's set


# ---------------------------------------------------------------------------
# System with mixer — integration tests
# ---------------------------------------------------------------------------

RF_FREQS = np.linspace(2.3e9, 2.5e9, 51)
LO_FREQ = 3e9
IF_FREQS_EXPECTED = RF_FREQS + LO_FREQ  # high sideband: ~5.3–5.5 GHz


@pytest.fixture
def upconvert_system():
    """LNA → Mixer (high sideband, 3 GHz LO) → PA"""
    lna = make_device(RF_FREQS, name="LNA", s21_db=15.0, nf_db=2.0, p1db_dbm=10.0)
    mixer = make_mixer(
        RF_FREQS,
        lo_freq=LO_FREQ,
        sideband="high",
        name="MXR",
        conversion_loss_db=7.0,
    )
    pa = make_device(RF_FREQS, name="PA", s21_db=30.0, nf_db=5.0, p1db_dbm=40.0)
    return System(name="UPC", manufacturer="Test", pn="SYS-001", devices=[lna, mixer, pa])


class TestSystemWithMixer:
    def test_device_count(self, upconvert_system):
        assert len(upconvert_system.devices) == 3

    def test_pre_mixer_device_not_translated(self, upconvert_system):
        lna = upconvert_system.devices[0]
        assert not lna.freq_plan.is_translated

    def test_mixer_device_is_translated(self, upconvert_system):
        mixer = upconvert_system.devices[1]
        assert mixer.freq_plan.is_translated

    def test_post_mixer_device_is_translated(self, upconvert_system):
        pa = upconvert_system.devices[2]
        assert pa.freq_plan.is_translated

    def test_mixer_carrier_is_upconverted(self, upconvert_system):
        mixer = upconvert_system.devices[1]
        np.testing.assert_allclose(
            mixer.freq_plan.carrier,
            IF_FREQS_EXPECTED,
            rtol=1e-6,
        )

    def test_post_mixer_carrier_matches_mixer(self, upconvert_system):
        mixer = upconvert_system.devices[1]
        pa = upconvert_system.devices[2]
        np.testing.assert_array_equal(pa.freq_plan.carrier, mixer.freq_plan.carrier)

    def test_rf_index_never_changes(self, upconvert_system):
        """All devices share the same RF index."""
        rf_ref = upconvert_system.devices[0].freq_plan.rf
        for d in upconvert_system.devices:
            np.testing.assert_array_equal(d.freq_plan.rf, rf_ref)

    def test_get_device_attr_has_carrier_freq_coord(self, upconvert_system):
        gain = upconvert_system.get_device_attr("gain")
        assert "carrier_freq" in gain.coords

    def test_carrier_freq_coord_matches_output_band(self, upconvert_system):
        gain = upconvert_system.get_device_attr("gain")
        # Last device (PA) should have carrier_freq in the IF band
        pa_carrier = gain.carrier_freq.isel(device=-1).values
        assert pa_carrier.min() > LO_FREQ, "Post-mixer carrier should be above LO"

    def test_system_network_frequency_axis_is_output_carrier(self, upconvert_system):
        """System.network frequency axis should be the output (carrier) frequencies."""
        net_freqs = upconvert_system.network.f
        np.testing.assert_allclose(net_freqs, IF_FREQS_EXPECTED, rtol=1e-6)

    def test_cascaded_gain_has_correct_shape(self, upconvert_system):
        cg = upconvert_system.cascaded_gain
        assert cg.sizes["device"] == 3
        assert cg.sizes["frequency"] == len(RF_FREQS)

    def test_gain_approx_correct(self, upconvert_system):
        """Total gain ≈ LNA(15) + Mixer(-7) + PA(30) = 38 dB."""
        gain = upconvert_system.gain
        gain_db = gain.data.to("dB").magnitude
        np.testing.assert_allclose(gain_db, 38.0, atol=0.5)

    def test_get_dataframe_frequency_column_is_carrier(self, upconvert_system):
        df = upconvert_system.get_dataframe(
            frequency=RF_FREQS[25],
            method="nearest",
        )
        # get_dataframe replaces frequency with carrier_freq, check first and last stages being less
        # than and greater than the LO
        freq_val = df.data["frequency"].iloc[0]
        assert freq_val < LO_FREQ
        freq_val = df.data["frequency"].iloc[-1]
        assert freq_val > LO_FREQ


# ---------------------------------------------------------------------------
# Downconvert system
# ---------------------------------------------------------------------------


@pytest.fixture
def downconvert_system():
    """LNA → Mixer (low sideband, 10 GHz LO) at X-band input → IF output"""
    xband_freqs = np.linspace(9.5e9, 10.5e9, 51)
    lo = 10.0e9
    lna = make_device(xband_freqs, name="LNA", s21_db=20.0, nf_db=1.5)
    mixer = make_mixer(xband_freqs, lo_freq=lo, sideband="low", name="MXR")
    if_amp = make_device(xband_freqs, name="IFA", s21_db=20.0, nf_db=3.0)
    return System(name="DNC", manufacturer="Test", pn="SYS-002", devices=[lna, mixer, if_amp])


class TestDownconvertSystem:
    def test_mixer_carrier_is_downconverted(self, downconvert_system):
        mixer = downconvert_system.devices[1]
        xband_freqs = np.linspace(9.5e9, 10.5e9, 51)
        lo = 10.0e9
        expected_if = np.abs(xband_freqs - lo)  # 0–500 MHz
        np.testing.assert_allclose(mixer.freq_plan.carrier, expected_if, rtol=1e-6)

    def test_post_mixer_carrier_in_if_band(self, downconvert_system):
        if_amp = downconvert_system.devices[2]
        assert if_amp.freq_plan.carrier.max() < 1e9, "IF carrier should be sub-GHz"

    def test_rf_index_still_xband(self, downconvert_system):
        for d in downconvert_system.devices:
            assert d.freq_plan.rf.min() >= 9e9
