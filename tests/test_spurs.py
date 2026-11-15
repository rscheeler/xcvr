"""Tests for mixer spur analysis (xcvr.spurs) and its integration with Mixer / System."""

import matplotlib as mpl
import numpy as np
import pytest
import xarray as xr
from xrench.units import ureg

mpl.use("Agg")
import matplotlib.pyplot as plt

from xcvr import System
from xcvr.devices import Constant, MixerConstant
from xcvr.spurs import (
    SpurTable,
    box_crossings,
    compute_spurs,
    fmt_freq,
    freq_unit,
    plot_spectrum,
    plot_spur_chart,
    plot_spur_sweep,
    plot_table,
    print_report,
    spur_free,
    spur_free_ranges,
    sweep_spurs,
)

MHZ = 1e6
GHZ = 1e9


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def table():
    return SpurTable.default()


# --------------------------------------------------------------------- table
class TestSpurTable:
    def test_level_inside_table(self, table):
        assert table.level(1, 1) == 0.0
        assert table.level(3, 2) == 85.0

    def test_level_nan_and_outside_fall_back_to_floor(self, table):
        assert table.level(0, 0) == table.floor_dbc  # NaN entry
        assert table.level(50, 1) == table.floor_dbc
        assert table.level(1, 50) == table.floor_dbc

    def test_shape_properties_and_repr(self, table):
        assert (table.max_m, table.max_n) == (5, 5)
        assert "illustrative" in repr(table)

    def test_from_csv(self, tmp_path):
        f = tmp_path / "t.csv"
        f.write_text("xx,45,75\n30,0,70\n\n60,50,xx\n")
        t = SpurTable.from_csv(str(f), floor_dbc=90)
        assert np.isnan(t.dbc[0, 0])
        assert t.level(1, 1) == 0.0
        assert t.level(2, 2) == 90  # "xx" -> floor
        assert t.name == str(f)


# ------------------------------------------------------------- single point
class TestComputeSpurs:
    def test_readme_modulated_example(self, table):
        """README: RF 200, LO 270, 10 MHz wide -> 3RF-2LO is 33% in band at -89.8 dBc."""
        sp = compute_spurs(
            200 * MHZ,
            270 * MHZ,
            table,
            max_order=9,
            in_bw=10 * MHZ,
            out_center=70 * MHZ,
        )
        inband = {(s.m, s.n, s.sign): s for s in sp if s.in_band and not s.is_desired}
        assert set(inband) == {(3, 2, -1), (5, 4, -1)}
        s32 = inband[(3, 2, -1)]
        assert s32.overlap_frac == pytest.approx(1 / 3, abs=0.01)
        assert s32.eff_dbc == pytest.approx(89.8, abs=0.1)
        assert s32.f_low == pytest.approx(45 * MHZ)
        assert s32.f_high == pytest.approx(75 * MHZ)

    def test_accepts_quantities(self, table):
        a = compute_spurs(2.4 * ureg.GHz, 1.8 * ureg.GHz, table, in_bw=40 * ureg.MHz)
        b = compute_spurs(2.4e9, 1.8e9, table, in_bw=40e6)
        assert [(s.m, s.n, s.sign, s.in_band) for s in a] == [
            (s.m, s.n, s.sign, s.in_band) for s in b
        ]

    def test_desired_flagged_and_sorted(self, table):
        sp = compute_spurs(900 * MHZ, 1000 * MHZ, table, out_bw=10 * MHZ)
        assert sum(s.is_desired for s in sp) == 1
        eff = [s.eff_dbc for s in sp]
        assert eff == sorted(eff)
        assert sp[0].label.startswith("|")

    def test_cw_desired_lands_on_band_centre(self, table):
        sp = compute_spurs(900 * MHZ, 1000 * MHZ, table)  # CW, zero-width band
        d = next(s for s in sp if s.is_desired)
        assert d.in_band
        assert d.eff_dbc == 0.0

    def test_backoff_improves_higher_order(self, table):
        a = compute_spurs(900 * MHZ, 1000 * MHZ, table)
        b = compute_spurs(900 * MHZ, 1000 * MHZ, table, rf_backoff_db=10)
        da = {(s.m, s.n, s.sign): s.dbc for s in a}
        db = {(s.m, s.n, s.sign): s.dbc for s in b}
        assert db[(3, 2, -1)] - da[(3, 2, -1)] == pytest.approx(20)  # (m-1)*10
        assert db[(1, 2, 1)] == da[(1, 2, 1)]  # m=1 unchanged

    def test_report(self, table, capsys):
        sp = compute_spurs(200 * MHZ, 270 * MHZ, table, max_order=9, in_bw=10 * MHZ)
        print_report(sp, 200 * MHZ, 270 * MHZ, 70 * MHZ, 10 * MHZ)
        out = capsys.readouterr().out
        assert "Worst in-band spur" in out
        assert "89.8" in out

    def test_report_clean(self, table, capsys):
        sp = compute_spurs(900 * MHZ, 1000 * MHZ, table, max_order=3, out_bw=1 * MHZ)
        print_report(sp, 900 * MHZ, 1000 * MHZ, 100 * MHZ, 1 * MHZ)
        assert "No spurs fall inside" in capsys.readouterr().out


# -------------------------------------------------------------------- sweep
class TestSweepSpurs:
    f = np.linspace(2.3 * GHZ, 2.5 * GHZ, 41)

    def test_matches_pointwise_compute_spurs(self, table):
        bw = 20 * MHZ
        sw = sweep_spurs(table, self.f, 3 * GHZ, desired_sign=1, signal_bw=bw, max_order=9)
        for i, fi in enumerate(self.f):
            ref = {
                (s.m, s.n, s.sign): s
                for s in compute_spurs(
                    fi, 3 * GHZ, table, max_order=9, desired_sign=1, out_bw=bw, in_bw=bw
                )
                if not s.is_desired
            }
            for k in range(sw.sizes["spur"]):
                s = ref[(int(sw.m[k]), int(sw.n[k]), int(sw.sign[k]))]
                assert bool(sw.in_band[i, k]) == s.in_band
                assert float(sw.eff_dbc[i, k]) == pytest.approx(s.eff_dbc)
                assert float(sw.f_low[i, k]) == pytest.approx(s.f_low)
                assert float(sw.f_high[i, k]) == pytest.approx(s.f_high)

    def test_structure(self, table):
        sw = sweep_spurs(table, self.f, 3 * GHZ, desired_sign=1, signal_bw=20 * MHZ, spec_dbc=60)
        assert sw.attrs["units"] == "Hz"
        assert sw.attrs["spec_dbc"] == 60
        np.testing.assert_allclose(sw["out_center"], self.f + 3 * GHZ)
        assert {"m", "n", "sign", "label"} <= set(sw.coords)
        # the desired product is not one of the spurs
        assert not ((sw.m == 1) & (sw.n == 1) & (sw.sign == 1)).any()

    def test_swept_lo_array(self, table):
        lo = self.f + 500 * MHZ  # tracking LO -> fixed IF
        sw = sweep_spurs(table, self.f, lo, desired_sign=-1, signal_bw=10 * MHZ)
        np.testing.assert_allclose(sw["out_center"], 500 * MHZ)
        np.testing.assert_allclose(sw["lo"], lo)

    def test_worst_is_nan_when_clean(self, table):
        sw = sweep_spurs(table, [2.0 * GHZ], 1.8 * GHZ, desired_sign=-1, signal_bw=1 * MHZ)
        assert sw["worst_dbc"].isnull().all()

    def test_spur_free_and_ranges(self, table):
        f = np.linspace(2.0 * GHZ, 2.6 * GHZ, 1201)
        sw = sweep_spurs(
            table, f, 1.8 * GHZ, desired_sign=-1, signal_bw=40 * MHZ, max_order=9, spec_dbc=88
        )
        ok = spur_free(sw)
        assert not ok.all()  # the 2x3 spur at 2.4 GHz violates -88 dBc
        ranges = spur_free_ranges(sw)
        assert len(ranges) == 2
        (a0, a1), (b0, b1) = ranges
        assert a0 == pytest.approx(2.0 * GHZ)
        assert b1 == pytest.approx(2.6 * GHZ)
        assert a1 < 2.4 * GHZ < b0
        # a looser spec removes the hole
        assert len(spur_free_ranges(sw, spec_dbc=70)) == 1

    def test_overlap_fraction_changes_judgement(self, table):
        """2x3 is -80 dBc but only 50% inside the band -> 83 dB of in-band power."""
        f = np.linspace(2.0 * GHZ, 2.6 * GHZ, 1201)
        kw = dict(desired_sign=-1, signal_bw=40 * MHZ, max_order=9, spec_dbc=82)
        cons = sweep_spurs(table, f, 1.8 * GHZ, **kw)
        est = sweep_spurs(table, f, 1.8 * GHZ, use_overlap_fraction=True, **kw)
        k = int(np.where((cons.m == 2) & (cons.n == 3) & (cons.sign == -1))[0][0])
        near = int(np.argmin(np.abs(f - 2.4 * GHZ)))
        assert float(cons.level[near, k]) == 80.0  # full table level
        assert float(est.level[near, k]) == pytest.approx(83.0, abs=0.05)
        assert len(spur_free_ranges(cons)) == 2  # conservative: hole at 2.4 GHz
        assert len(spur_free_ranges(est)) == 1  # estimate: 83 dB >= 82 dB everywhere
        assert est.attrs["use_overlap_fraction"] == 1

    def test_plot_and_spur_free_agree(self, table):
        """Red segments in the plot are exactly where spur_free() says the spec fails."""
        f = np.linspace(2.0 * GHZ, 2.6 * GHZ, 1201)
        sw = sweep_spurs(
            table, f, 1.8 * GHZ, desired_sign=-1, signal_bw=40 * MHZ, max_order=9, spec_dbc=82
        )
        ax1, _ = plot_spur_sweep(sw)
        red = [ln for ln in ax1.lines if ln.get_color() == "red"]
        assert len(red) == 1
        x_red = np.asarray(red[0].get_xdata(), dtype=float)
        y_red = np.asarray(red[0].get_ydata(), dtype=float)
        x_fail = x_red[~np.isnan(y_red)] * 1e9  # axis is in GHz
        bad = f[~spur_free(sw).to_numpy()]
        assert x_fail.min() == pytest.approx(bad.min(), abs=1e6)
        assert x_fail.max() == pytest.approx(bad.max(), abs=1e6)

    def test_spur_free_needs_spec(self, table):
        sw = sweep_spurs(table, self.f, 3 * GHZ, desired_sign=1, signal_bw=20 * MHZ)
        with pytest.raises(ValueError, match="No spec"):
            spur_free(sw)


# --------------------------------------------------------------------- box
def test_box_crossings(table):
    rows = box_crossings(table, 1000, (850, 950), (50, 150), desired_sign=-1, max_order=9)
    assert rows == sorted(rows, key=lambda r: r["dbc"])
    assert all(r["y_high"] > 50 and r["y_low"] < 150 for r in rows)
    assert not any((r["m"], r["n"], r["sign"]) == (1, 1, -1) for r in rows)  # desired excluded


# ------------------------------------------------------------------- units
def test_unit_helpers():
    assert freq_unit(2.4e9) == (1e9, "GHz")
    assert freq_unit(70e6) == (1e6, "MHz")
    assert freq_unit(5e3) == (1e3, "kHz")
    assert freq_unit(10.0) == (1.0, "Hz")
    assert fmt_freq(250e6) == "250 MHz"


# -------------------------------------------------------------------- plots
class TestPlots:
    def test_table_and_spectrum(self, table):
        plot_table(table)
        sp = compute_spurs(200 * MHZ, 270 * MHZ, table, max_order=9, in_bw=10 * MHZ)
        ax = plot_spectrum(sp, 200 * MHZ, 270 * MHZ, 70 * MHZ, 10 * MHZ)
        assert "MHz" in ax.get_xlabel()

    def test_spur_chart_with_signal_box(self, table):
        ax = plot_spur_chart(
            table,
            11 * GHZ,
            in_range=(1 * GHZ, 3 * GHZ),
            out_range=(12 * GHZ, 14 * GHZ),
            desired_sign=1,
            signal_bw=250 * MHZ,
            signal_center=1.5 * GHZ,
            spec_dbc=60,
        )
        assert "GHz" in ax.get_xlabel()
        assert "signal box" in ax.get_title()

    def test_spur_chart_defaults_and_errors(self, table):
        plot_spur_chart(table, 1000 * MHZ, (850 * MHZ, 950 * MHZ))  # out_range derived
        with pytest.raises(ValueError, match="needs signal_bw"):
            plot_spur_chart(table, 1000 * MHZ, (850 * MHZ, 950 * MHZ), signal_center=900 * MHZ)
        with pytest.raises(ValueError, match="outside both"):
            plot_spur_chart(
                table,
                1000 * MHZ,
                (850 * MHZ, 950 * MHZ),
                signal_bw=10 * MHZ,
                signal_center=5 * GHZ,
            )

    @pytest.mark.parametrize("x", ["input", "output", "lo"])
    def test_spur_sweep_plot(self, table, x):
        f = np.linspace(2.0 * GHZ, 2.6 * GHZ, 601)
        sw = sweep_spurs(
            table, f, 1.8 * GHZ, desired_sign=-1, signal_bw=40 * MHZ, max_order=9, spec_dbc=88
        )
        if x == "lo":  # needs a varying LO to be meaningful
            sw = sweep_spurs(table, f, f - 500 * MHZ, desired_sign=-1, signal_bw=40 * MHZ)
        ax1, ax2 = plot_spur_sweep(sw, x=x)
        assert ax1.get_ylabel().endswith("(MHz)") or ax1.get_ylabel().endswith("(GHz)")
        assert ax2.get_xlabel().endswith("(GHz)")

    def test_spur_sweep_plot_without_spec(self, table):
        f = np.linspace(2.0 * GHZ, 2.6 * GHZ, 301)
        sw = sweep_spurs(table, f, 1.8 * GHZ, desired_sign=-1, signal_bw=40 * MHZ)
        plot_spur_sweep(sw, y_range=(0, 1.2 * GHZ), title="t", label_dbc=None)


# ----------------------------------------------------------- mixer / system
FREQ = np.linspace(2.0e9, 2.6e9, 61)


def _frequency():
    return xr.DataArray(FREQ * ureg.Hz, dims=("frequency",), coords={"frequency": FREQ})


def _mixer(table=None, lo=1.8 * ureg.GHz, sideband="low", name="MX1"):
    return MixerConstant(
        name,
        "Test",
        "MX",
        frequency=_frequency(),
        gain=-7 * ureg.dB,
        nf=7 * ureg.dB,
        lo_freq=lo,
        sideband=sideband,
        spur_table=table,
    )


class TestMixerSpurs:
    def test_desired_sign_follows_sideband(self, table):
        assert _mixer(table, sideband="high").desired_sign == 1
        assert _mixer(table, sideband="low").desired_sign == -1

    def test_spur_sweep_matches_standalone(self, table):
        mx = _mixer(table)
        sw = mx.spur_sweep(FREQ * ureg.Hz, 40 * ureg.MHz, max_order=9, spec_dbc=88, npts=1201)
        ref = sweep_spurs(
            table,
            np.linspace(2.0e9, 2.6e9, 1201),
            1.8e9,
            desired_sign=-1,
            signal_bw=40e6,
            max_order=9,
            spec_dbc=88,
        )
        np.testing.assert_allclose(sw["worst_dbc"], ref["worst_dbc"], equal_nan=True)

    def test_spur_sweep_resamples_sparse_grid(self, table):
        sw = _mixer(table).spur_sweep(FREQ, 40e6, npts=500)
        assert sw.sizes["frequency"] == 500

    def test_point_spurs(self, table):
        sp = _mixer(table).spurs(2.4 * ureg.GHz, 40 * ureg.MHz, max_order=9)
        assert any(s.in_band and not s.is_desired and (s.m, s.n) == (2, 3) for s in sp)

    def test_requires_table(self):
        with pytest.raises(ValueError, match="no spur_table"):
            _mixer(None).spur_sweep(FREQ, 40e6)
        with pytest.raises(ValueError, match="no spur_table"):
            _mixer(None).spurs(2.4e9)

    def test_multistage_collapsed_mixer_rejected(self, table):
        mx = _mixer(table)
        mx.stages = [(1.8 * ureg.GHz, "low"), (1.0 * ureg.GHz, "high")]
        with pytest.raises(ValueError, match="2 mixer stages"):
            mx.spur_sweep(FREQ, 40e6)


class TestSystemSpurSweeps:
    def _system(self, table, with_table=True):
        f = _frequency()
        lna = Constant("LNA", "Test", "1", frequency=f, gain=20 * ureg.dB, nf=2 * ureg.dB)
        mx = _mixer(table if with_table else None)
        ifa = Constant("IFA", "Test", "3", frequency=f, gain=15 * ureg.dB, nf=3 * ureg.dB)
        return System("RX", "Test", "SYS", [lna, mx, ifa])

    def test_sweeps_use_carrier_at_mixer_input(self, table):
        sw = self._system(table).spur_sweeps(40 * ureg.MHz, spec_dbc=88)
        assert list(sw) == ["MX1"]
        d = sw["MX1"]
        assert float(d.frequency.min()) == pytest.approx(2.0e9)
        assert float(d.frequency.max()) == pytest.approx(2.6e9)
        np.testing.assert_allclose(d["lo"], 1.8e9)
        assert d.attrs["desired_sign"] == -1
        ranges = spur_free_ranges(d)
        assert len(ranges) == 2  # hole around 2.4 GHz (2x3 spur)

    def test_mixers_without_table_are_skipped(self, table):
        assert self._system(table, with_table=False).spur_sweeps(40e6) == {}

    def test_second_mixer_sees_translated_carrier(self, table):
        f = _frequency()
        m1 = _mixer(table, lo=1.8 * ureg.GHz, sideband="low", name="M1")  # 2.0-2.6 -> 0.2-0.8 GHz
        m2 = _mixer(table, lo=1.0 * ureg.GHz, sideband="high", name="M2")  # -> 1.2-1.8 GHz
        pad = Constant("PAD", "Test", "1", frequency=f, gain=-3 * ureg.dB, nf=3 * ureg.dB)
        sysm = System("RX2", "Test", "S2", [m1, pad, m2])
        sw = sysm.spur_sweeps(10 * ureg.MHz)
        assert set(sw) == {"M1", "M2"}
        assert float(sw["M2"].frequency.min()) == pytest.approx(0.2e9)
        assert float(sw["M2"].frequency.max()) == pytest.approx(0.8e9)
        assert sw["M2"].attrs["desired_sign"] == 1
