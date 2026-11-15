"""Tests for LO frequency planning (xcvr.loplan)."""

import matplotlib as mpl
import numpy as np
import pytest
from xrench.units import ureg

mpl.use("Agg")
import matplotlib.pyplot as plt

from xcvr.loplan import (
    Converter,
    analyze,
    best_los,
    filter_report,
    plot_filter_mask,
    plot_lo_sweep,
    plot_spur_map,
    sweep_lo,
)
from xcvr.spurs import SpurTable

GHZ, MHZ = 1e9, 1e6


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def up():
    return Converter(
        "IF->Ku up",
        1 * GHZ,
        3 * GHZ,
        "up",
        "upper",
        spec_dbc=60,
        signal_bw=250 * MHZ,
        rf_window=(12 * GHZ, 16 * GHZ),
        table=SpurTable.default(),
    )


@pytest.fixture
def dn():
    return Converter(
        "Ku->IF dn",
        1 * GHZ,
        3 * GHZ,
        "down",
        "lower",
        spec_dbc=60,
        signal_bw=250 * MHZ,
        rf_window=(8 * GHZ, 13.5 * GHZ),
        table=SpurTable.default(),
    )


class TestConverter:
    def test_quantities_are_stored_in_hz(self):
        c = Converter(
            "q",
            1 * ureg.GHz,
            3000 * ureg.MHz,
            signal_bw=0.25 * ureg.GHz,
            rf_window=(12 * ureg.GHz, 16 * ureg.GHz),
        )
        assert (c.if_lo, c.if_hi, c.signal_bw) == (1e9, 3e9, 250e6)
        assert c.rf_window == (12e9, 16e9)
        assert c.bw == 2e9
        assert c.sig_bw == 250e6

    def test_defaults(self):
        c = Converter("d", 1e9, 3e9, spec_dbc=55)
        assert c.sig_bw == c.bw  # signal fills the band
        assert c.pb_spec == 55

    @pytest.mark.parametrize("kw", [{"direction": "sideways"}, {"sideband": "middle"}])
    def test_validation(self, kw):
        with pytest.raises(ValueError, match="must be"):
            Converter("bad", 1e9, 3e9, **kw)

    def test_geometry(self, up, dn):
        g = up.geometry(11 * GHZ)
        assert g["rf"] == (12 * GHZ, 14 * GHZ) and g["sign"] == 1 and g["ok"]
        assert not up.geometry(13.5 * GHZ)["ok"]  # RF 14.5-16.5 GHz leaves the window
        assert up.geometry(13 * GHZ)["ok"]  # RF 14-16 GHz just fits
        assert dn.geometry(2 * GHZ) is None  # lower sideband needs LO > IF
        assert dn.geometry(13 * GHZ)["sign"] == -1


class TestAnalysis:
    def test_readme_filter_requirements(self, up):
        """README example: LO = 11 GHz up-converter (illustrative table)."""
        an = analyze(up, 11 * GHZ)
        need = {
            (i.spur.m, i.spur.n, i.spur.sign): round(i.rejection, 1)
            for i in an.infos
            if i.rejection > 0
        }
        assert need[(1, 1, -1)] == 60.0  # image, plain mixer
        assert need[(0, 1, 1)] == 30.0  # LO leakage
        assert need[(1, 2, -1)] == 10.0
        assert an.guard == pytest.approx(1 * GHZ)  # LO leak 1 GHz below the passband
        assert an.max_rej == 60.0
        assert an.margin > 0  # nothing inside the passband

    def test_image_rejection_reduces_requirement(self):
        kw = dict(spec_dbc=60, table=SpurTable.default())
        plain = analyze(Converter("p", 1e9, 3e9, "up", "upper", **kw), 11e9)
        ssb = analyze(Converter("s", 1e9, 3e9, "up", "upper", image_rejection_db=25, **kw), 11e9)
        assert plain.max_rej - ssb.max_rej == pytest.approx(25)

    def test_infeasible_lo(self, dn):
        an = analyze(dn, 2 * GHZ)
        assert an.geo is None and an.infos == []

    def test_guard_at(self, up):
        an = analyze(up, 11 * GHZ)
        assert an.guard_at(60) == pytest.approx(2 * GHZ)  # image 2 GHz away
        assert an.guard_at(25) == pytest.approx(1 * GHZ)

    def test_mask(self, up):
        an = analyze(up, 11 * GHZ)
        off = np.linspace(-6 * GHZ, 8 * GHZ, 1401)
        req, bad = an.mask(off)
        assert req.max() == 60.0 and bad.max() == 0.0
        inside = (off >= 0) & (off <= up.bw)
        assert (req[inside] == 0).all()  # nothing to filter inside the passband

    def test_in_passband_violation_is_unfilterable(self):
        """Weak table: a 2x1 spur lands in the passband and fails -> cannot be filtered."""
        t = SpurTable(np.array([[np.nan, 30, 30], [30, 0, 30], [30, 30, 30]]), floor_dbc=30)
        c = Converter("x", 1e9, 2e9, "up", "upper", spec_dbc=60, table=t)
        sw = sweep_lo(c, np.arange(5e9, 9e9, 50e6))[0]
        assert (~sw.valid & sw.allowed).any()
        assert any(a.guard == 0 for a in (analyze(c, lo) for lo in sw.lo))

    def test_filter_report_text(self, up, dn, capsys):
        filter_report(analyze(up, 11 * GHZ))
        out = capsys.readouterr().out
        assert "LO = 11 GHz" in out and "signal 250 MHz" in out
        assert "60.0 dB" in out
        filter_report(analyze(dn, 2 * GHZ))
        assert "not feasible" in capsys.readouterr().out


class TestQuantityLO:
    """Every public entry point accepts the LO as a pint Quantity as well as Hz."""

    def test_analyze_and_report(self, up, capsys):
        a, b = analyze(up, 11 * ureg.GHz), analyze(up, 11 * GHZ)
        assert a.lo == b.lo == 11 * GHZ
        assert a.max_rej == b.max_rej and a.guard == b.guard
        filter_report(a)
        assert "LO = 11 GHz" in capsys.readouterr().out
        plot_filter_mask(a)

    def test_analyze_infeasible_quantity(self, dn):
        assert analyze(dn, 2 * ureg.GHz).geo is None

    def test_spur_map(self, up):
        plot_spur_map(up, 11 * ureg.GHz, signal_center=1.5 * ureg.GHz)


class TestSweep:
    def test_sweep_and_best_los(self, up):
        sw = sweep_lo(up, np.arange(9 * GHZ, 14 * GHZ, 5 * MHZ))[0]
        # RF window 12-16 GHz and upper sideband -> LO feasible for 11..13 GHz
        assert sw.lo[sw.allowed].min() == pytest.approx(11 * GHZ)
        assert sw.lo[sw.allowed].max() <= 13 * GHZ
        assert sw.valid[sw.allowed].all()
        wins = best_los(sw, verbose=False)
        assert len(wins) == 1
        assert wins[0]["best_lo"] == pytest.approx(11 * GHZ)
        assert wins[0]["guard"] == pytest.approx(1 * GHZ)

    def test_quantity_lo_values(self, up):
        a = sweep_lo(up, np.arange(9000, 14000, 5.0) * ureg.MHz)[0]
        b = sweep_lo(up, np.arange(9 * GHZ, 14 * GHZ, 5 * MHZ))[0]
        np.testing.assert_array_equal(a.valid, b.valid)

    def test_shared_lo_prints_windows(self, up, capsys):
        sw = sweep_lo([up, up], np.arange(10 * GHZ, 14 * GHZ, 10 * MHZ))
        assert len(best_los(sw)) == 1
        assert "Valid LO windows" in capsys.readouterr().out

    def test_no_common_lo_explains_why(self, up, dn, capsys):
        sw = sweep_lo([up, dn], np.arange(9 * GHZ, 11 * GHZ, 10 * MHZ))
        assert best_los(sw) == []
        out = capsys.readouterr().out
        assert "No LO in the swept range" in out


class TestPlots:
    def test_lo_sweep_plot(self, up, dn):
        sw = sweep_lo([up, dn], np.arange(9 * GHZ, 14 * GHZ, 20 * MHZ))
        fig = plot_lo_sweep(sw)
        assert any("GHz" in ax.get_xlabel() for ax in fig.axes)
        plot_lo_sweep(sw[0], zoom=False)

    def test_filter_mask_plot(self, up):
        ax = plot_filter_mask(analyze(up, 11 * GHZ))
        assert "GHz" in ax.get_xlabel() and "11 GHz" in ax.get_title()

    def test_spur_map_plot(self, up):
        ax = plot_spur_map(up, 11 * GHZ, signal_center=1.5 * GHZ)
        assert "GHz" in ax.get_xlabel()
        with pytest.raises(ValueError, match="not feasible"):
            plot_spur_map(Converter("d", 1e9, 3e9, "down", "lower"), 2 * GHZ)
