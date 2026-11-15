"""Tests for tone tracing (xcvr.spurtrace)."""

from types import SimpleNamespace

import matplotlib as mpl
import numpy as np
import pytest
import skrf
import xarray as xr
from xrench.units import ureg

mpl.use("Agg")
import matplotlib.pyplot as plt

from xcvr import System, ToneSource, plot_spectrum_sweep
from xcvr.devices import Constant, MixerConstant
from xcvr.spurs import SpurTable
from xcvr.spurtrace import trace_tones
from xcvr.xcvr import Device

GRID = np.linspace(2.0e9, 2.6e9, 31)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _freq():
    return xr.DataArray(GRID * ureg.Hz, dims=("frequency",), coords={"frequency": GRID})


def _amp(name, gain):
    return Constant(name, "T", "A", frequency=_freq(), gain=gain * ureg.dB, nf=3 * ureg.dB)


def _mixer(name="MX1", lo=1.8, sideband="low", table=True, **kw):
    return MixerConstant(
        name,
        "T",
        "M",
        frequency=_freq(),
        gain=-7 * ureg.dB,
        nf=7 * ureg.dB,
        lo_freq=lo * ureg.GHz,
        sideband=sideband,
        spur_table=SpurTable.default() if table else None,
        **kw,
    )


def _bandpass(name, f0, bw, il=1.0, order=3):
    f = np.linspace(0.05e9, 45e9, 900)
    g = np.maximum(-il - 3.0 * (np.abs(f - f0) / (bw / 2)) ** (2 * order), -90)
    s = np.zeros((f.size, 2, 2), complex)
    s[:, 1, 0] = s[:, 0, 1] = 10 ** (g / 20)
    net = skrf.Network(frequency=skrf.Frequency.from_f(f, unit="hz"), s=s)
    return Device(name, "T", "BPF", net, nf=ureg.Quantity(il, "dB")), (f, g)


def _by_path(tr, node=-1):
    ds = tr.dataset(node)
    return {str(ds.label[k].item()): k for k in range(ds.sizes["tone"])}, ds


@pytest.fixture
def rx():
    return System("RX", "T", "1", [_amp("LNA", 20), _mixer()])


class TestPowers:
    def test_hand_calculated_levels(self, rx):
        """-30 dBm + 20 dB LNA - 7 dB mixer = -17 dBm desired; others from the table."""
        tr = rx.trace_tones(ToneSource(np.array([2.4e9]), -30 * ureg.dBm), max_order=9)
        paths, ds = _by_path(tr)
        p = lambda path: float(ds.power_dbm[0, paths[path]])  # noqa: E731
        f = lambda path: float(ds.freq[0, paths[path]])  # noqa: E731
        assert (f("RF > MX1:1x1-"), p("RF > MX1:1x1-")) == (pytest.approx(0.6e9), -17.0)
        assert ds.desired[paths["RF > MX1:1x1-"]]
        assert (f("RF > MX1:1x1+"), p("RF > MX1:1x1+")) == (pytest.approx(4.2e9), -17.0)  # image
        assert p("LO > MX1:0x1") == pytest.approx(-47.0)  # 30 dBc LO leakage
        assert f("LO > MX1:0x1") == pytest.approx(1.8e9)
        assert p("RF > MX1:1x0+") == pytest.approx(-62.0)  # 45 dBc RF feedthrough
        assert p("RF > MX1:1x2-") == pytest.approx(-67.0)  # table[n=2][m=1] = 50 dBc
        assert f("RF > MX1:1x2-") == pytest.approx(1.2e9)

    def test_2x3_lands_on_the_desired_at_2p4ghz(self, rx):
        tr = rx.trace_tones(ToneSource(np.array([2.4e9]), -30 * ureg.dBm), max_order=9)
        paths, ds = _by_path(tr)
        k = paths["RF > MX1:2x3-"]
        assert float(ds.freq[0, k]) == pytest.approx(0.6e9)  # same frequency as the desired
        assert float(ds.power_dbm[0, k]) == pytest.approx(-17 - 80)

    def test_swept_source_traces(self, rx):
        sw = np.linspace(2.0e9, 2.6e9, 13)
        tr = rx.trace_tones(ToneSource(sw, -30 * ureg.dBm), max_order=5)
        paths, ds = _by_path(tr)
        assert ds.sizes["sweep"] == 13
        np.testing.assert_allclose(ds.sweep, sw)
        np.testing.assert_allclose(ds.freq[:, paths["RF > MX1:1x1-"]], sw - 1.8e9)
        np.testing.assert_allclose(ds.freq[:, paths["RF > MX1:2x1-"]], np.abs(2 * sw - 1.8e9))

    def test_nodes(self, rx):
        tr = rx.trace_tones(ToneSource(2.4e9, -30 * ureg.dBm))
        assert tr.node_names == ["input", "LNA", "MX1"]
        # node 1 is after the LNA: single tone at -10 dBm
        d1 = tr.dataset(1)
        assert d1.sizes["tone"] == 1 and float(d1.power_dbm[0, 0]) == pytest.approx(-10.0)
        assert tr.dataset("LNA").attrs["node"] == 1
        assert tr.dataset(0).sizes["tone"] == 1
        with pytest.raises(KeyError, match="No node"):
            tr.dataset("nope")

    def test_ambiguous_name(self):
        a = _amp("AMP", 10)
        tr = System("S", "T", "1", [a, a]).trace_tones(ToneSource(2.4e9, 0))
        with pytest.raises(KeyError, match="ambiguous"):
            tr.dataset("AMP")
        assert tr.dataset(2).attrs["node"] == 2


class TestModelDetails:
    def test_filter_gain_comes_from_native_sparams_out_of_grid(self):
        """The filter data covers 0.05-45 GHz; the cascade grid only 2-2.6 GHz."""
        bpf, (f, g) = _bandpass("IFBPF", 0.6e9, 0.4e9)
        sysm = System("RX", "T", "1", [_amp("LNA", 20), _mixer(), bpf])
        tr = sysm.trace_tones(ToneSource(np.array([2.4e9]), -30 * ureg.dBm), max_order=9)
        paths, ds = _by_path(tr)
        for path, fq in (("RF > MX1:1x1-", 0.6e9), ("RF > MX1:1x1+", 4.2e9)):
            k = paths[path]
            expect = -17.0 + np.interp(fq, f, g)
            assert float(ds.power_dbm[0, k]) == pytest.approx(expect, abs=1e-6)
        # the image is rejected by ~60+ dB, the desired passes with its insertion loss
        assert float(ds.power_dbm[0, paths["RF > MX1:1x1+"]]) < -75
        assert float(ds.power_dbm[0, paths["RF > MX1:1x1-"]]) == pytest.approx(-18.0, abs=0.01)

    def test_spur_ref_power_moves_higher_order_products(self):
        base = System("S", "T", "1", [_mixer()])
        ref = System("S", "T", "1", [_mixer(spur_ref_power=-30 * ureg.dBm)])
        src = ToneSource(np.array([2.4e9]), -40 * ureg.dBm)  # 10 dB below the reference
        a = base.trace_tones(src, max_order=9)
        b = ref.trace_tones(src, max_order=9)
        pa, da = _by_path(a)
        pb, db = _by_path(b)
        d = lambda ds, paths, path: float(ds.power_dbm[0, paths[path]])  # noqa: E731
        assert d(db, pb, "RF > MX1:1x1-") == d(da, pa, "RF > MX1:1x1-")  # desired unchanged
        assert d(db, pb, "RF > MX1:1x2-") == d(da, pa, "RF > MX1:1x2-")  # m = 1 unchanged
        assert d(db, pb, "RF > MX1:2x1-") - d(da, pa, "RF > MX1:2x1-") == pytest.approx(-10)
        assert d(db, pb, "RF > MX1:3x1-") - d(da, pa, "RF > MX1:3x1-") == pytest.approx(-20)

    def test_mixer_without_table_passes_only_desired(self):
        sysm = System("S", "T", "1", [_amp("LNA", 20), _mixer(table=False)])
        tr = sysm.trace_tones(ToneSource(np.array([2.4e9]), -30 * ureg.dBm))
        ds = tr.dataset()
        assert ds.sizes["tone"] == 1
        assert float(ds.freq[0, 0]) == pytest.approx(0.6e9)
        assert float(ds.power_dbm[0, 0]) == pytest.approx(-17.0)

    def test_high_sideband_and_two_mixers(self):
        m1 = _mixer("M1", lo=1.8, sideband="low")  # 2.4 -> 0.6
        m2 = _mixer("M2", lo=1.0, sideband="high", table=False)  # 0.6 -> 1.6
        tr = System("S", "T", "1", [m1, m2]).trace_tones(
            ToneSource(np.array([2.4e9]), -30 * ureg.dBm)
        )
        paths, ds = _by_path(tr)
        k = paths["RF > M1:1x1- > M2:1x1"]
        assert float(ds.freq[0, k]) == pytest.approx(1.6e9)
        assert float(ds.power_dbm[0, k]) == pytest.approx(-30 - 7 - 7)
        assert ds.desired[k]

    def test_nested_system_equals_flat(self):
        flat = System("F", "T", "1", [_amp("LNA", 20), _mixer(), _amp("IFA", 10)])
        nest = System(
            "N",
            "T",
            "2",
            [_amp("LNA", 20), System("BE", "T", "3", [_mixer(), _amp("IFA", 10)])],
        )
        src = ToneSource(np.linspace(2.2e9, 2.4e9, 5), -30 * ureg.dBm)
        a, b = flat.trace_tones(src, max_order=5), nest.trace_tones(src, max_order=5)
        np.testing.assert_allclose(a.dataset().power_dbm, b.dataset().power_dbm)
        np.testing.assert_allclose(a.dataset().freq, b.dataset().freq)

    def test_multiple_sources_and_lo_once(self, rx):
        sw = np.linspace(2.0e9, 2.6e9, 7)
        srcs = [
            ToneSource(sw, -30 * ureg.dBm, "RF"),
            ToneSource(2.3e9, -50 * ureg.dBm, "JAM"),  # fixed interferer
        ]
        tr = rx.trace_tones(srcs, max_order=3)
        paths, ds = _by_path(tr)
        assert "JAM > MX1:1x1-" in paths and "RF > MX1:1x1-" in paths
        assert float(ds.power_dbm[0, paths["JAM > MX1:1x1-"]]) == pytest.approx(-37.0)
        assert sum(p.startswith("LO > MX1:0x1") for p in paths) == 1  # not once per source
        assert tr.sweep_name == "RF"

    def test_pruning(self, rx):
        src = ToneSource(np.array([2.4e9]), -30 * ureg.dBm)
        full = rx.trace_tones(src, max_order=9, min_power_dbm=-300).dataset()
        cut = rx.trace_tones(src, max_order=9, min_power_dbm=-60).dataset()
        capped = rx.trace_tones(src, max_order=9, max_tones=3).dataset()
        assert cut.sizes["tone"] < full.sizes["tone"]
        assert (cut.peak_dbm >= -60).all()
        assert capped.sizes["tone"] == 3
        assert capped.peak_dbm.max() == full.peak_dbm.max()

    def test_validation(self, rx):
        with pytest.raises(ValueError, match="same number of points"):
            rx.trace_tones(
                [ToneSource(np.linspace(2e9, 2.5e9, 5)), ToneSource(np.linspace(2e9, 2.5e9, 7))]
            )

    def test_multistage_mixer_without_table(self):
        mx = _mixer(table=False)
        mx.stages = [(1.8 * ureg.GHz, "low"), (1.0 * ureg.GHz, "high")]
        tr = System("S", "T", "1", [mx]).trace_tones(ToneSource(np.array([2.4e9]), -30))
        assert float(tr.dataset().freq[0, 0]) == pytest.approx(1.6e9)

    def test_multistage_mixer_with_table_rejected(self):
        mx = _mixer()
        mx.stages = [(1.8 * ureg.GHz, "low"), (1.0 * ureg.GHz, "high")]
        with pytest.raises(ValueError, match="several mixers"):
            trace_tones(System("S", "T", "1", [mx]), ToneSource(2.4e9, -30))


class TestOutputs:
    def test_summary(self, rx):
        tr = rx.trace_tones(ToneSource(np.linspace(2.2e9, 2.6e9, 9), -30 * ureg.dBm), max_order=5)
        df = tr.summary(n=4)
        assert len(df) == 4
        assert df.peak_dBm.is_monotonic_decreasing
        assert {"path", "desired", "peak_dBm", "sweep_at_peak_Hz"} <= set(df.columns)

    def test_empty_dataset_and_plot(self, rx):
        tr = rx.trace_tones(ToneSource(2.4e9, -300), min_power_dbm=-100)
        ds = tr.dataset()
        assert ds.sizes["tone"] == 0
        assert "No tones" in plot_spectrum_sweep(ds).get_title()

    def test_plot_and_hover(self):
        bpf, _ = _bandpass("RFBPF", 2.3e9, 1.0e9)
        sysm = System("RX", "T", "1", [_amp("LNA", 20), bpf, _mixer()])
        tr = sysm.trace_tones(
            ToneSource(np.linspace(1.0e9, 4.0e9, 121), -30 * ureg.dBm), max_order=7
        )
        ds = tr.dataset(3)
        ax = plot_spectrum_sweep(ds, n_markers=3)
        assert "node 3" in ax.get_title() and ax.get_xlabel().startswith("Frequency (GHz")
        # exercise the tooltip on the strongest point
        fig = ax.figure
        k = int(np.argmax(ds.peak_dbm.to_numpy()))
        j = int(np.argmax(ds.power_dbm[:, k].to_numpy()))
        x, y = ax.transData.transform((ds.freq[j, k] / 1e9, ds.power_dbm[j, k]))

        ev = SimpleNamespace(inaxes=ax, x=float(x), y=float(y))  # minimal matplotlib event
        fig._spur_hover(ev)
        ann = [
            c
            for c in ax.get_children()
            if c.__class__.__name__ == "Annotation"
            and c.get_visible()
            and "dBm" in c.get_text()
            and "RF =" in c.get_text()
        ]
        assert ann
        ev.x, ev.y = float(x) + 400, float(y) + 400  # far from any point -> hidden
        fig._spur_hover(ev)
        assert not any(
            c.__class__.__name__ == "Annotation"
            and c.get_visible()
            and "dBm" in c.get_text()
            and "RF =" in c.get_text()
            for c in ax.get_children()
        )
        ev.inaxes = None
        fig._spur_hover(ev)
        plt.close(fig)


class TestAgreesWithCascade:
    """The desired path must reproduce xcvr's own cascaded gain, point for point."""

    @staticmethod
    def _dev(cls, name, f, gain_db, **kw):
        s = np.zeros((f.size, 2, 2), complex)
        s[:, 1, 0] = s[:, 0, 1] = 10 ** (gain_db / 20)
        net = skrf.Network(frequency=skrf.Frequency.from_f(f, unit="hz"), s=s)
        return cls(name, "T", name, net, nf=ureg.Quantity(3, "dB"), **kw)

    def _chain(self):
        from xcvr.devices import Mixer

        rf = np.linspace(2.3e9, 2.5e9, 21)
        f = np.linspace(0.05e9, 6e9, 1200)  # dense native data -> interpolation error ~0
        table = SpurTable.default()
        lna = self._dev(Device, "LNA", rf, 20 + 4 * (rf - 2.3e9) / 0.2e9)  # tilts with RF
        m1 = self._dev(
            Mixer,
            "M1",
            f,
            -6 - 2 * f / 6e9,
            lo_freq=1.8 * ureg.GHz,
            sideband="low",
            spur_table=table,
        )
        ifbpf = self._dev(Device, "IFBPF", f, -1 - 8 * ((f - 0.6e9) / 1e9) ** 2)  # at the IF
        m2 = self._dev(
            Mixer,
            "M2",
            f,
            -10 + 7 * np.clip((f - 0.6e9) / 1.8e9, 0, 1),
            lo_freq=1.0 * ureg.GHz,
            sideband="high",
            spur_table=table,
        )
        ifa = self._dev(Device, "IFA", f, 15 - 3 * (f / 6e9))
        return System("S", "T", "1", [lna, m1, ifbpf, m2, ifa]), rf

    def test_desired_path_equals_system_gain(self):
        sysm, rf = self._chain()
        gain = np.asarray(sysm.gain.data.to("dB").magnitude, dtype=float)
        tr = sysm.trace_tones(ToneSource(rf, -40 * ureg.dBm), max_order=5)
        ds = tr.dataset()
        k = int(np.where(ds.desired.to_numpy())[0][0])
        traced = ds.power_dbm[:, k].to_numpy() + 40  # gain = Pout - Pin
        np.testing.assert_allclose(traced, gain, atol=0.05)
        # and it lands where xcvr says: 2.4 GHz -> 0.6 GHz -> 1.6 GHz
        np.testing.assert_allclose(ds.freq[:, k], sysm.devices[-1].freq_plan.carrier, rtol=1e-9)

    def test_every_node_matches_cascaded_gain(self):
        sysm, rf = self._chain()
        cg = sysm.cascaded_gain  # (device, frequency)
        tr = sysm.trace_tones(ToneSource(rf, -40 * ureg.dBm), max_order=5)
        for node in range(1, 6):
            ds = tr.dataset(node)
            k = int(np.where(ds.desired.to_numpy())[0][0])
            expect = np.asarray(cg.isel(device=node - 1).data.to("dB").magnitude, dtype=float)
            np.testing.assert_allclose(ds.power_dbm[:, k].to_numpy() + 40, expect, atol=0.05)


# ----------------------------------------------------------------- trace checks
def _wide_amp(name, gain, p1db=None, nf=3.0):
    """Amp defined over 0.05-20 GHz so spurs stay inside its data."""
    g = np.linspace(0.05e9, 20e9, 80)
    fr = xr.DataArray(g * ureg.Hz, dims=("frequency",), coords={"frequency": g})
    kw = {} if p1db is None else {"p1db": p1db * ureg.dBm}
    return Constant(name, "T", "A", frequency=fr, gain=gain * ureg.dB, nf=nf * ureg.dB, **kw)


def _src(n=21, p=-30.0):
    return ToneSource(np.linspace(2.0e9, 2.6e9, n), p)


class TestExtrapolation:
    def test_flags_tones_outside_device_range(self):
        # amp covers only 2.0-2.6 GHz; the mixer's lower-sideband output (~0.2-0.8 GHz) and
        # any spur that leaves that range must be flagged at the amp after the mixer.
        s = System("S", "T", "1", [_mixer(), _amp("A2", 10)])
        ds = trace_tones(s, _src(), warn=False).dataset(-1)
        des = int(np.flatnonzero(ds.desired.to_numpy())[0])
        assert ds.extrapolated[:, des].all()
        assert "A2" in str(ds.extrap_devices[des].item())

    def test_inside_range_not_flagged(self):
        s = System("S", "T", "1", [_wide_amp("A1", 10), _wide_amp("A2", 10)])
        ds = trace_tones(s, _src(), warn=False).dataset(-1)
        assert not ds.extrapolated.to_numpy().any()
        assert (ds.extrap_devices.to_numpy() == "").all()

    def test_warning_and_summary_column(self):
        s = System("S", "T", "1", [_mixer(), _amp("A2", 10)])
        with pytest.warns(UserWarning, match="outside its S-parameter range"):
            tr = trace_tones(s, _src())
        cols = tr.summary(-1).columns
        assert "extrapolated_in" in cols
        assert tr.summary(-1).extrapolated_pct.iloc[0] == 100.0

    def test_plot_draws_hollow_markers(self):
        s = System("S", "T", "1", [_mixer(), _amp("A2", 10)])
        ds = trace_tones(s, _src(), warn=False).dataset(-1)
        ax = plot_spectrum_sweep(ds, hover=False)
        assert any(line.get_marker() == "o" and line.get_mfc() == "none" for line in ax.lines)
        text = "".join(t.get_text() for t in ax.texts)
        assert "outside data" in text


class TestPruningReport:
    def test_counts_order_cap(self):
        s = System("S", "T", "1", [_wide_amp("A1", 10), _mixer()])
        full = trace_tones(s, _src(), max_order=None, warn=False).pruning
        capped = trace_tones(s, _src(), max_order=3, warn=False).pruning
        mx_full = full[full.device == "MX1"].iloc[0]
        mx_cap = capped[capped.device == "MX1"].iloc[0]
        assert mx_full.dropped_order == 0
        assert mx_cap.dropped_order > 0
        assert mx_cap.kept + mx_cap.dropped_order >= mx_full.kept - 1
        assert np.isfinite(mx_cap.strongest_dropped_dbm)

    def test_max_tones_and_significance_warning(self):
        s = System("S", "T", "1", [_wide_amp("A1", 10), _mixer()])
        with pytest.warns(UserWarning, match="pruning removed"):
            tr = trace_tones(s, _src(), max_tones=3)
        row = tr.pruning.query("device == 'MX1'").iloc[0]
        assert row.kept == 3
        assert row.dropped_cap > 0
        assert row.significant

    def test_floor_drops_counted(self):
        s = System("S", "T", "1", [_wide_amp("A1", 10), _mixer()])
        row = trace_tones(s, _src(), min_power_dbm=-60, warn=False).pruning.iloc[1]
        assert row.dropped_floor > 0


class TestCompressionFlag:
    def test_flags_when_output_exceeds_p1db(self):
        s = System("S", "T", "1", [_wide_amp("A1", 20, p1db=0.0)])
        with pytest.warns(UserWarning, match="P1dB"):
            tr = trace_tones(s, ToneSource(np.linspace(2e9, 2.6e9, 5), -10.0))
        row = tr.compression.iloc[0]
        assert row.flagged
        assert row.max_out_dbm == pytest.approx(10.0, abs=0.1)
        assert row.margin_db == pytest.approx(-10.0, abs=0.1)

    def test_not_flagged_with_headroom_and_infinite_default(self):
        s = System("S", "T", "1", [_wide_amp("A1", 20, p1db=30.0), _wide_amp("A2", 5)])
        tr = trace_tones(s, ToneSource(np.linspace(2e9, 2.6e9, 5), -30.0), warn=False)
        assert not tr.compression.flagged.any()
        assert np.isinf(tr.compression.margin_db.iloc[1])

    def test_margin_threshold(self):
        s = System("S", "T", "1", [_wide_amp("A1", 20, p1db=0.0)])
        src = ToneSource(np.linspace(2e9, 2.6e9, 5), -25.0)  # output -5 dBm, 5 dB below P1dB
        assert not trace_tones(s, src, warn=False).compression.flagged.any()
        assert trace_tones(s, src, compression_margin_db=6, warn=False).compression.flagged.all()


class TestNoiseFloor:
    def test_friis_by_hand_and_node0(self):
        s = System("S", "T", "1", [_wide_amp("A1", 20, nf=3.0), _wide_amp("A2", 10, nf=6.0)])
        tr = trace_tones(s, _src(5), warn=False)
        k_t0 = 10 * np.log10(1.380649e-23 * 290 / 1e-3)
        assert tr.dataset(0).noise_dbm_hz.to_numpy() == pytest.approx(k_t0)
        f = 10**0.3 + (10**0.6 - 1) / 100
        expect = k_t0 + 10 * np.log10(f) + 30
        assert tr.dataset(-1).noise_dbm_hz.to_numpy() == pytest.approx(expect, abs=0.01)

    def test_matches_system_nf_and_gain(self):
        s = System("S", "T", "1", [_wide_amp("LNA", 20), _mixer(), _wide_amp("IFA", 10)])
        f = np.array([2.1e9, 2.3e9, 2.5e9])
        tr = trace_tones(s, ToneSource(f, -50.0), warn=False)
        ds = tr.dataset(-1)
        k_t0 = 10 * np.log10(1.380649e-23 * 290 / 1e-3)
        fr = s.nf.frequency.to_numpy()
        idx = [int(np.argmin(np.abs(fr - x))) for x in ds.sweep.to_numpy()]
        nf = s.nf.data.to("dB").magnitude[idx]
        g = s.gain.data.to("dB").magnitude[idx]
        assert ds.noise_dbm_hz.to_numpy() == pytest.approx(k_t0 + nf + g, abs=0.1)

    def test_noise_follows_desired_frequency_and_plot(self):
        s = System("S", "T", "1", [_mixer(), _wide_amp("A2", 10)])
        ds = trace_tones(s, _src(), warn=False).dataset(-1)
        des = int(np.flatnonzero(ds.desired.to_numpy())[0])
        assert ds.noise_freq.to_numpy() == pytest.approx(ds.freq[:, des].to_numpy())
        ax = plot_spectrum_sweep(ds, hover=False)
        assert any("noise floor" in t.get_text() for t in ax.texts)
        ax2 = plot_spectrum_sweep(ds, hover=False, show_noise=False)
        assert not any("noise floor" in t.get_text() for t in ax2.texts)
