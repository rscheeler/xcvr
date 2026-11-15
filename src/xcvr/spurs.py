"""
Mixer spur (m x n) analysis.

Computes the mixing products ``|m*IN +/- n*LO|`` of a mixer, attaches the datasheet
suppression (dBc below the desired output) for every product, and provides:

* :func:`compute_spurs`: all spurs for one input/LO pair (single point).
* :func:`sweep_spurs`: spurs while the input (and optionally the LO) is swept. Returns an
  :class:`xarray.Dataset` aligned with the rest of xcvr.
* :func:`spur_free_ranges`: the swept-input ranges where no spur violates a spec.
* :func:`box_crossings`: products passing through an input-band x output-band box.
* Plots: :func:`plot_table`, :func:`plot_spectrum`, :func:`plot_spur_chart` (classic chart) and
  :func:`plot_spur_sweep` (swept-input view with a spur-free strip).

Conventions
-----------
* The engine is unit-agnostic (all frequencies must simply share one unit). The public
  functions accept floats in **Hz** or pint ``Quantity`` objects and convert for you; plots
  auto-scale to Hz/kHz/MHz/GHz.
* ``desired_sign = -1`` means desired output = ``|IN - LO|`` (xcvr ``sideband="low"``),
  ``+1`` means ``IN + LO`` (xcvr ``sideband="high"``). This matches
  :meth:`xcvr.frequency.FrequencyPlan.translate`.
* Spur table rows are n (LO harmonic) and columns are m (input harmonic), values are dB
  below the desired 1x1 product. ``NaN`` means "not specified" (falls back to ``floor_dbc``).

The bundled default table is ILLUSTRATIVE ONLY. Replace it with your mixer's datasheet table.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr
from pint import Quantity

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = [
    "Spur",
    "SpurTable",
    "box_crossings",
    "compute_spurs",
    "plot_spectrum",
    "plot_spur_chart",
    "plot_spur_sweep",
    "plot_table",
    "print_report",
    "spur_free",
    "spur_free_ranges",
    "sweep_spurs",
]

# Relative tolerance for "a CW product lands exactly on the output band centre"
_CW_RTOL = 1e-12

# Illustrative double-balanced-mixer table. rows = n (LO), cols = m (input).
# (m=1, n=1) is the desired product (0 dBc).
_DEFAULT_DBC = np.array(
    [
        [np.nan, 45, 75, 85, 90, 95],
        [30, 0, 70, 82, 88, 92],
        [60, 50, 68, 85, 88, 95],
        [65, 72, 80, 78, 90, 95],
        [80, 80, 85, 90, 92, 95],
        [85, 85, 90, 92, 95, 98],
    ],
    dtype=float,
)


# ----------------------------------------------------------------------- units
def to_hz(value: float | Quantity | np.ndarray) -> Any:
    """Convert a Quantity (any frequency unit) to Hz; bare numbers are assumed to be Hz."""
    if isinstance(value, Quantity):
        return value.to("Hz").magnitude
    return value


def freq_unit(values: Any) -> tuple[float, str]:
    """Pick a display unit for frequencies (Hz): returns ``(divisor, name)``."""
    top = float(np.nanmax(np.abs(np.asarray(values, dtype=float)))) if np.size(values) else 0.0
    for div, name in ((1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz")):
        if top >= div:
            return div, name
    return 1.0, "Hz"


def fmt_freq(value: float, unit: tuple[float, str] | None = None) -> str:
    """Format a frequency in Hz with an auto-chosen (or given) unit."""
    div, name = unit or freq_unit(value)
    return f"{value / div:g} {name}"


# ----------------------------------------------------------------------- table
class SpurTable:
    """m x n spur suppression table (dBc below the desired product), indexed ``dbc[n][m]``."""

    def __init__(
        self,
        dbc: Any,
        name: str = "illustrative default",
        floor_dbc: float = 100.0,
    ) -> None:
        self.dbc = np.asarray(dbc, dtype=float)
        self.name = name
        self.floor_dbc = floor_dbc  # assumed level beyond the table

    def __repr__(self) -> str:
        return f"SpurTable({self.name!r}, {self.max_m}x{self.max_n}, floor={self.floor_dbc:g} dBc)"

    @property
    def max_m(self) -> int:
        """Largest input harmonic in the table."""
        return int(self.dbc.shape[1] - 1)

    @property
    def max_n(self) -> int:
        """Largest LO harmonic in the table."""
        return int(self.dbc.shape[0] - 1)

    def level(self, m: int, n: int) -> float:
        """Suppression (dBc) of the (m, n) product, or the floor if outside/unspecified."""
        if 0 <= n <= self.max_n and 0 <= m <= self.max_m:
            v = self.dbc[n, m]
            if not np.isnan(v):
                return float(v)
        return self.floor_dbc

    @classmethod
    def default(cls) -> SpurTable:
        """The bundled ILLUSTRATIVE table (replace with datasheet values)."""
        return cls(_DEFAULT_DBC.copy())

    @classmethod
    def from_csv(cls, path: str, **kw: Any) -> SpurTable:
        """Load a CSV (no header) where row i is n=i and column j is m=j."""
        rows = []
        with open(path, newline="") as fh:  # noqa: PTH123
            for r in csv.reader(fh):
                if not r:
                    continue
                rows.append(
                    [
                        float(c) if c.strip().lower() not in ("", "xx", "-", "nan") else np.nan
                        for c in r
                    ],
                )
        kw.setdefault("name", str(path))
        return cls(np.array(rows, dtype=float), **kw)


# ---------------------------------------------------------------- single point
@dataclass(frozen=True)
class Spur:
    """One mixing product at one operating point."""

    m: int
    n: int
    sign: int  # freq = |m*IN + sign*n*LO|
    freq: float
    dbc: float  # dB below desired (larger = better)
    is_desired: bool
    in_band: bool
    f_low: float = 0.0
    f_high: float = 0.0
    overlap_frac: float = 1.0  # fraction of the spur power inside the output band

    @property
    def eff_dbc(self) -> float:
        """Suppression of the power that actually lands in the output band."""
        if self.is_desired or self.overlap_frac <= 0:
            return self.dbc
        return float(self.dbc - 10 * np.log10(self.overlap_frac))

    @property
    def order(self) -> int:
        """m + n."""
        return self.m + self.n

    @property
    def label(self) -> str:
        """Human-readable product name."""
        s = "+" if self.sign > 0 else "-"
        return f"|{self.m}IN {s} {self.n}LO|"


def _products(table: SpurTable, max_order: int | None):
    """Yield every (m, n, sign) product in the table, excluding (0, 0)."""
    for m in range(table.max_m + 1):
        for n in range(table.max_n + 1):
            if (m == 0 and n == 0) or (max_order is not None and m + n > max_order):
                continue
            # With m == 0 or n == 0 both signs give the same frequency.
            for s in (1,) if (m == 0 or n == 0) else (1, -1):
                yield m, n, s


def compute_spurs(
    f_in: float | Quantity,
    f_lo: float | Quantity,
    table: SpurTable,
    max_order: int | None = None,
    rf_backoff_db: float = 0.0,
    desired_sign: int = -1,
    out_center: float | Quantity | None = None,
    out_bw: float | Quantity = 0.0,
    in_bw: float | Quantity = 0.0,
) -> list[Spur]:
    """Spurs for one input/LO pair, worst (smallest dBc) first.

    Args:
        f_in: Mixer input frequency (Hz).
        f_lo: LO frequency (Hz).
        table: Spur table.
        max_order: Ignore products with m + n above this.
        rf_backoff_db: Input drive below the table's reference level. An (m x n) product
            falls m dB per dB of drive reduction while the desired falls 1 dB, so dBc
            improves by ``(m - 1) * backoff``.
        desired_sign: -1 for ``|IN - LO|``, +1 for ``IN + LO``.
        out_center: Centre of the output band to protect (default: desired product).
        out_bw: Full width of that band (default: ``in_bw``).
        in_bw: Modulated input bandwidth. An m*IN product occupies ``m * in_bw``; a spur is
            "in band" if its band overlaps the output band (flat-spectrum approximation).
    """
    f_in, f_lo = float(to_hz(f_in)), float(to_hz(f_lo))
    in_bw, out_bw = float(to_hz(in_bw)), float(to_hz(out_bw))
    if out_bw == 0.0 and in_bw > 0.0:
        out_bw = in_bw
    desired_f = abs(f_in + desired_sign * f_lo)
    centre = desired_f if out_center is None else float(to_hz(out_center))
    half = out_bw / 2.0
    out = []
    for m, n, s in _products(table, max_order):
        f = abs(m * f_in + s * n * f_lo)
        desired = m == 1 and n == 1 and s == desired_sign
        dbc = 0.0 if desired else table.level(m, n) + (m - 1) * rf_backoff_db
        a = m * (f_in - in_bw / 2) + s * n * f_lo
        b = m * (f_in + in_bw / 2) + s * n * f_lo
        lo_x, hi_x = min(a, b), max(a, b)
        f_low = 0.0 if lo_x <= 0 <= hi_x else min(abs(lo_x), abs(hi_x))
        f_high = max(abs(lo_x), abs(hi_x))
        ov = min(f_high, centre + half) - max(f_low, centre - half)
        width = f_high - f_low
        if width <= 1e-12 * max(f_high, 1.0):  # CW product
            in_band = abs(f - centre) <= half + _CW_RTOL * max(centre, 1.0)
            frac = 1.0 if in_band else 0.0
        else:
            in_band = ov > 0
            frac = max(ov, 0.0) / width
        out.append(Spur(m, n, s, f, dbc, desired, in_band, f_low, f_high, frac))
    out.sort(key=lambda sp: sp.eff_dbc)
    return out


def print_report(
    spurs: list[Spur],
    f_in: float | Quantity,
    f_lo: float | Quantity,
    out_center: float | Quantity,
    out_bw: float | Quantity,
) -> None:
    """Print the in-band spurs of a :func:`compute_spurs` result."""
    f_in, f_lo = float(to_hz(f_in)), float(to_hz(f_lo))
    out_center, out_bw = float(to_hz(out_center)), float(to_hz(out_bw))
    u = freq_unit([f_in, f_lo, out_center, out_bw])
    print(
        f"IN = {fmt_freq(f_in, u)}, LO = {fmt_freq(f_lo, u)}, "
        f"output band = {fmt_freq(out_center, u)} +/- {fmt_freq(out_bw / 2, u)}",
    )
    bad = [s for s in spurs if s.in_band and not s.is_desired]
    if not bad:
        print("No spurs fall inside the output band.")
        return
    print(f"{'product':<14}{'order':>6}{'band (' + u[1] + ')':>26}{'in-band':>9}{'dBc':>8}")
    for s in bad:
        band = f"{s.f_low / u[0]:.3f}-{s.f_high / u[0]:.3f}"
        print(f"{s.label:<14}{s.order:>6}{band:>26}{s.overlap_frac * 100:>8.0f}%{-s.eff_dbc:>8.1f}")
    w = min(bad, key=lambda s: s.eff_dbc)
    print(f"Worst in-band spur: {w.label} at {-w.eff_dbc:.1f} dBc")


# ------------------------------------------------------------------- sweeping
def sweep_spurs(
    table: SpurTable,
    f_in: Any,
    f_lo: Any,
    desired_sign: int = -1,
    signal_bw: float | Quantity = 0.0,
    out_bw: float | Quantity | None = None,
    max_order: int | None = None,
    rf_backoff_db: float = 0.0,
    spec_dbc: float | None = None,
    use_overlap_fraction: bool = False,
) -> xr.Dataset:
    """Spurs of a mixer while the input frequency (and optionally the LO) is swept.

    At every sweep point the signal (width ``signal_bw``) is centred on ``f_in``; the output
    band to protect is the desired product +/- ``out_bw / 2``. This is the swept view
    behind :func:`plot_spur_sweep`.

    Args:
        table: Spur table.
        f_in: Input centre frequencies (Hz or Quantity array).
        f_lo: LO frequency: a scalar (fixed LO) or an array the same length as ``f_in``
            (e.g. a tracking LO that keeps a fixed IF).
        desired_sign: -1 for ``|IN - LO|``, +1 for ``IN + LO``.
        signal_bw: Instantaneous input signal bandwidth (0 = CW).
        out_bw: Output band width (default: ``signal_bw``).
        max_order: Ignore products with m + n above this.
        rf_backoff_db: Input drive below the table's reference level.
        spec_dbc: Optional spur spec stored in the result attrs (used by the plot).
        use_overlap_fraction: How an in-band spur is judged. False (default, conservative):
            at its full table level, however little of its band overlaps the output band.
            True: at the level of the power that actually lands in the band (flat-spectrum
            estimate; see ``eff_dbc``). This is the same choice as
            ``Converter(use_overlap_fraction=...)`` in :mod:`xcvr.loplan`.

    Returns:
        Dataset with dims ``frequency`` (the swept input, Hz) and ``spur`` (one per product):

        * ``out_center(frequency)``: desired output frequency.
        * ``f_center, f_low, f_high (frequency, spur)``: spur band edges.
        * ``dbc(spur)``: suppression below desired, with backoff applied.
        * ``eff_dbc(frequency, spur)``: suppression of the power landing in the output band.
        * ``level(frequency, spur)``: the level spurs are judged at (``dbc`` or ``eff_dbc``,
          per ``use_overlap_fraction``).
        * ``in_band(frequency, spur)``: spur band overlaps the output band.
        * ``worst_dbc(frequency)``: worst in-band ``level`` (NaN if nothing is in band).
        * coords ``m, n, sign, label`` on ``spur``; ``lo(frequency)``.
    """
    f = np.atleast_1d(np.asarray(to_hz(f_in), dtype=float))
    lo = np.broadcast_to(np.asarray(to_hz(f_lo), dtype=float), f.shape)
    bw = float(to_hz(signal_bw))
    obw = bw if out_bw is None else float(to_hz(out_bw))
    half_o = obw / 2.0
    out_c = np.abs(f + desired_sign * lo)

    terms = [
        (m, n, s)
        for m, n, s in _products(table, max_order)
        if not (m == 1 and n == 1 and s == desired_sign)
    ]
    nf, nk = f.size, len(terms)
    f_center = np.empty((nf, nk))
    f_low = np.empty((nf, nk))
    f_high = np.empty((nf, nk))
    in_band = np.zeros((nf, nk), dtype=bool)
    eff = np.empty((nf, nk))
    dbc = np.empty(nk)
    for k, (m, n, s) in enumerate(terms):
        a = m * (f - bw / 2) + s * n * lo
        b = m * (f + bw / 2) + s * n * lo
        lo_x, hi_x = np.minimum(a, b), np.maximum(a, b)
        straddle = (lo_x <= 0) & (hi_x >= 0)
        y_lo = np.where(straddle, 0.0, np.minimum(np.abs(lo_x), np.abs(hi_x)))
        y_hi = np.maximum(np.abs(lo_x), np.abs(hi_x))
        centre = np.abs(m * f + s * n * lo)
        ov = np.minimum(y_hi, out_c + half_o) - np.maximum(y_lo, out_c - half_o)
        width = y_hi - y_lo
        cw = width <= 1e-12 * np.maximum(y_hi, 1.0)
        inb = np.where(
            cw, np.abs(centre - out_c) <= half_o + _CW_RTOL * np.maximum(out_c, 1.0), ov > 0
        )
        frac = np.where(cw, inb.astype(float), np.clip(ov, 0, None) / np.where(cw, 1.0, width))
        dbc[k] = table.level(m, n) + (m - 1) * rf_backoff_db
        with np.errstate(divide="ignore"):
            eff[:, k] = np.where(inb & (frac > 0), dbc[k] - 10 * np.log10(frac), dbc[k])
        f_center[:, k], f_low[:, k], f_high[:, k], in_band[:, k] = centre, y_lo, y_hi, inb

    level = eff if use_overlap_fraction else np.broadcast_to(dbc, eff.shape).copy()
    worst = np.where(in_band, level, np.inf).min(axis=1) if nk else np.full(nf, np.inf)
    worst = np.where(np.isfinite(worst), worst, np.nan)

    labels = [f"{m}IN{'+' if s > 0 else '-'}{n}LO" for m, n, s in terms]
    return xr.Dataset(
        data_vars={
            "out_center": ("frequency", out_c),
            "f_center": (("frequency", "spur"), f_center),
            "f_low": (("frequency", "spur"), f_low),
            "f_high": (("frequency", "spur"), f_high),
            "dbc": ("spur", dbc),
            "eff_dbc": (("frequency", "spur"), eff),
            "level": (("frequency", "spur"), level),
            "in_band": (("frequency", "spur"), in_band),
            "worst_dbc": ("frequency", worst),
        },
        coords={
            "frequency": f,
            "spur": np.arange(nk),
            "m": ("spur", [t[0] for t in terms]),
            "n": ("spur", [t[1] for t in terms]),
            "sign": ("spur", [t[2] for t in terms]),
            "label": ("spur", labels),
            "lo": ("frequency", np.array(lo)),
        },
        attrs={
            "units": "Hz",
            "desired_sign": desired_sign,
            "signal_bw": bw,
            "out_bw": obw,
            "floor_dbc": table.floor_dbc,
            "spec_dbc": np.nan if spec_dbc is None else spec_dbc,
            "table": table.name,
            "rf_backoff_db": rf_backoff_db,
            "use_overlap_fraction": int(use_overlap_fraction),
        },
    )


def spur_free(sweep: xr.Dataset, spec_dbc: float | None = None) -> xr.DataArray:
    """Boolean DataArray: True where no in-band spur is stronger than ``spec_dbc``.

    ``spec_dbc`` defaults to the value stored by :func:`sweep_spurs`.
    """
    spec = sweep.attrs["spec_dbc"] if spec_dbc is None else spec_dbc
    if spec is None or np.isnan(spec):
        raise ValueError("No spec: pass spec_dbc here or to sweep_spurs.")
    return (sweep["worst_dbc"].isnull() | (sweep["worst_dbc"] >= spec)).rename("spur_free")


def spur_free_ranges(
    sweep: xr.Dataset,
    spec_dbc: float | None = None,
) -> list[tuple[float, float]]:
    """Contiguous swept-input ranges (Hz) where the spec is met, as ``(start, stop)``."""
    ok = spur_free(sweep, spec_dbc).to_numpy()
    f = sweep["frequency"].to_numpy()
    out: list[tuple[float, float]] = []
    start = None
    for i, good in enumerate(ok):
        if good and start is None:
            start = i
        if start is not None and (not good or i == len(ok) - 1):
            end = i if good else i - 1
            out.append((float(f[start]), float(f[end])))
            start = None
    return out


# ------------------------------------------------------------------------ box
def box_crossings(
    table: SpurTable,
    f_lo: float,
    in_range: tuple[float, float],
    out_range: tuple[float, float],
    desired_sign: int = -1,
    max_order: int | None = None,
    rf_backoff_db: float = 0.0,
) -> list[dict]:
    """Products that pass through the box (input band x output band).

    A product crosses the box if, for some input frequency in ``in_range``, its output
    lands in ``out_range``. Those are the in-band spurs a filter cannot remove. Unit-agnostic
    (use one consistent unit). Sorted worst (smallest dBc) first.
    """
    in_lo, in_hi = in_range
    out_lo, out_hi = out_range
    rows = []
    for m, n, s in _products(table, max_order):
        if m == 1 and n == 1 and s == desired_sign:
            continue
        a, b = m * in_lo + s * n * f_lo, m * in_hi + s * n * f_lo
        lo_x, hi_x = min(a, b), max(a, b)
        y_lo = 0.0 if lo_x <= 0 <= hi_x else min(abs(lo_x), abs(hi_x))
        y_hi = max(abs(lo_x), abs(hi_x))
        if y_hi > out_lo and y_lo < out_hi:
            rows.append(
                {
                    "m": m,
                    "n": n,
                    "sign": s,
                    "y_low": y_lo,
                    "y_high": y_hi,
                    "dbc": table.level(m, n) + (m - 1) * rf_backoff_db,
                },
            )
    rows.sort(key=lambda r: r["dbc"])
    return rows


# ---------------------------------------------------------------------- plots
def plot_table(table: SpurTable, ax: Axes | None = None) -> Axes:
    """Heatmap of the m x n spur chart."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 5.5))
    data = np.ma.masked_invalid(table.dbc)
    im = ax.imshow(data, cmap="RdYlGn", origin="upper")
    for n in range(table.dbc.shape[0]):
        for m in range(table.dbc.shape[1]):
            v = table.dbc[n, m]
            ax.text(m, n, "xx" if np.isnan(v) else f"{v:.0f}", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(table.dbc.shape[1]))
    ax.set_yticks(range(table.dbc.shape[0]))
    ax.set_xlabel("m  (input harmonic)")
    ax.set_ylabel("n  (LO harmonic)")
    ax.set_title(f"Mixer spur chart (dBc)\n{table.name}")
    ax.figure.colorbar(im, ax=ax, label="suppression (dB below desired)")
    return ax


def plot_spectrum(
    spurs: list[Spur],
    f_in: float | Quantity,
    f_lo: float | Quantity,
    out_center: float | Quantity,
    out_bw: float | Quantity,
    ax: Axes | None = None,
    max_dbc: float | None = None,
) -> Axes:
    """Spur spectrum for one input/LO pair, with the output band highlighted."""
    import matplotlib.pyplot as plt

    f_in, f_lo = float(to_hz(f_in)), float(to_hz(f_lo))
    out_center, out_bw = float(to_hz(out_center)), float(to_hz(out_bw))
    div, un = freq_unit([f_in, f_lo, out_center])
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5))
    shown = [s for s in spurs if max_dbc is None or s.dbc <= max_dbc]
    for s in shown:
        color = "tab:green" if s.is_desired else ("tab:red" if s.in_band else "tab:blue")
        if s.f_high - s.f_low > 1e-12:
            ax.hlines(-s.dbc, s.f_low / div, s.f_high / div, color=color, lw=6, alpha=0.5)
        ax.vlines(s.freq / div, -100, -s.dbc, color=color, lw=1.0)
        if s.is_desired or s.in_band or s.dbc < 60:
            ax.annotate(
                f"{s.m}x{s.n}",
                (s.freq / div, -s.dbc),
                fontsize=7,
                ha="center",
                va="bottom",
                rotation=90,
            )
    if out_bw > 0:
        ax.axvspan(
            (out_center - out_bw / 2) / div,
            (out_center + out_bw / 2) / div,
            color="gold",
            alpha=0.3,
            label="output band",
        )
        ax.legend(loc="lower right")
    ax.set_ylim(-100, 8)
    ax.set_xlabel(f"Output frequency ({un})")
    ax.set_ylabel("Level relative to desired (dBc)")
    ax.set_title(
        f"Spur spectrum: IN={fmt_freq(f_in)}, LO={fmt_freq(f_lo)} (red = in-band spur)",
    )
    ax.grid(alpha=0.3)
    return ax


def _signal_input_center(
    center: Any,
    in_rng: tuple[float, float],
    out_rng: tuple[float, float],
    f_lo: float,
    sign: int,
) -> float:
    """Signal centre as an INPUT frequency (``center`` may be input- or output-side)."""
    in_lo, in_hi = in_rng
    out_lo, out_hi = out_rng
    side = None
    if isinstance(center, (tuple, list)):
        side, center = center
    if side is None:
        in_ok, out_ok = in_lo <= center <= in_hi, out_lo <= center <= out_hi
        if not (in_ok or out_ok):
            raise ValueError("signal_center is outside both the input range and the output range")
        side = "in" if in_ok else "out"
    if side == "in":
        return float(center)
    cands = [v for v in (center - sign * f_lo, -center - sign * f_lo) if in_lo <= v <= in_hi]
    if not cands:
        raise ValueError("output frequency is not reachable from the input range at this LO")
    return float(cands[0])


def plot_spur_chart(  # noqa: C901, PLR0912, PLR0915
    table: SpurTable,
    f_lo: float | Quantity,
    in_range: tuple[Any, Any],
    out_range: tuple[Any, Any] | None = None,
    desired_sign: int = -1,
    signal_bw: float | Quantity | None = None,
    signal_center: Any = None,
    spec_dbc: float | None = None,
    max_order: int | None = None,
    rf_backoff_db: float = 0.0,
    show_dbc: float | None = None,
    label_dbc: float | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Classic mixer spur chart: x = input frequency, y = output frequency.

    Every product ``|m*x +/- n*LO|`` is a line labelled ``mxn  -level`` and coloured by its
    suppression (red = strong, green = weak). The blue strip is the input range, the gold
    strip the output range, and their intersection (black outline) is the *box*. The thick
    black line is the desired product. A product crossing the box is an in-band spur
    (dashed if it meets ``spec_dbc``, bold red if it fails).

    Args:
        table: Spur table.
        f_lo: LO frequency (Hz or Quantity).
        in_range: ``(low, high)`` input range.
        out_range: ``(low, high)`` output range (default: where the desired product lands).
        desired_sign: -1 for ``|in - LO|``, +1 for ``in + LO``.
        signal_bw: Width of the signal inside the ranges; draws a purple signal box.
        signal_center: Where that signal sits: an input- or output-side frequency, or
            ``("in", f)`` / ``("out", f)`` to force a side. Default: middle of ``in_range``.
        spec_dbc: Spur spec; products in the box stronger than this are drawn bold red.
        max_order: Ignore products with m + n above this.
        rf_backoff_db: Input drive below the table's reference level.
        show_dbc: Hide products weaker than this (products in the box are always shown).
        label_dbc: Label only products at or below this suppression (default 75).
        ax: Axes to draw into.
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    f_lo_hz = float(to_hz(f_lo))
    in_hz = tuple(float(to_hz(v)) for v in in_range)
    if out_range is None:
        a, b = in_hz[0] + desired_sign * f_lo_hz, in_hz[1] + desired_sign * f_lo_hz
        out_hz = tuple(sorted((abs(a), abs(b))))
    else:
        out_hz = tuple(float(to_hz(v)) for v in out_range)
    div, un = freq_unit([in_hz[1], out_hz[1], f_lo_hz])

    # Work in display units (the engine is linear in frequency)
    f_lo = f_lo_hz / div
    in_lo, in_hi = (v / div for v in in_hz)
    out_lo, out_hi = (v / div for v in out_hz)
    if signal_bw is not None:
        signal_bw = float(to_hz(signal_bw)) / div
    if signal_center is not None:
        if isinstance(signal_center, (tuple, list)):
            signal_center = (signal_center[0], float(to_hz(signal_center[1])) / div)
        else:
            signal_center = float(to_hz(signal_center)) / div

    bw_in, bw_out = in_hi - in_lo, out_hi - out_lo
    x0, x1 = max(0.0, in_lo - bw_in), in_hi + bw_in
    y0, y1 = max(0.0, out_lo - bw_out), out_hi + bw_out
    if show_dbc is None:
        show_dbc = (spec_dbc + 20) if spec_dbc is not None else 80
    if label_dbc is None:
        label_dbc = min(show_dbc, 75)

    if signal_center is not None and signal_bw is None:
        raise ValueError("signal_center needs signal_bw")
    sig_in = sig_out = None
    sig_cross: dict = {}
    if signal_bw is not None and signal_bw < bw_in * (1 - 1e-9):
        c = (
            (in_lo + in_hi) / 2
            if signal_center is None
            else _signal_input_center(
                signal_center,
                (in_lo, in_hi),
                (out_lo, out_hi),
                f_lo,
                desired_sign,
            )
        )
        c = min(max(c, in_lo + signal_bw / 2), in_hi - signal_bw / 2)
        sig_in = (c - signal_bw / 2, c + signal_bw / 2)
        a, b = sig_in[0] + desired_sign * f_lo, sig_in[1] + desired_sign * f_lo
        sig_out = tuple(sorted((abs(a), abs(b))))
        sig_cross = {
            (r["m"], r["n"], r["sign"]): r
            for r in box_crossings(
                table,
                f_lo,
                sig_in,
                sig_out,
                desired_sign,
                max_order,
                rf_backoff_db,
            )
        }
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 7.5))

    cross = box_crossings(
        table,
        f_lo,
        (in_lo, in_hi),
        (out_lo, out_hi),
        desired_sign,
        max_order,
        rf_backoff_db,
    )
    crossing = {(r["m"], r["n"], r["sign"]): r for r in cross}

    ax.axvspan(in_lo, in_hi, color="tab:blue", alpha=0.12, lw=0, zorder=0)
    ax.axhspan(out_lo, out_hi, color="gold", alpha=0.30, lw=0, zorder=0)

    xs = np.linspace(x0, x1, 1201)
    box_labels = []
    norm = colors.Normalize(vmin=0, vmax=show_dbc)
    cmap = cm.RdYlGn
    for m, n, s in _products(table, max_order):
        if m == 1 and n == 1 and s == desired_sign:
            continue
        dbc = table.level(m, n) + (m - 1) * rf_backoff_db
        crosses = (m, n, s) in crossing or (m, n, s) in sig_cross
        if dbc > show_dbc and not crosses:
            continue
        y = np.abs(m * xs + s * n * f_lo)
        inview = (y >= y0) & (y <= y1)
        if not inview.any():
            continue
        fail = crosses and spec_dbc is not None and dbc < spec_dbc
        if fail:
            ax.plot(xs, y, color="red", lw=3.0, zorder=6)
        elif crosses:
            ax.plot(xs, y, color=cmap(norm(dbc)), lw=2.2, ls="--", zorder=5)
        else:
            ax.plot(xs, y, color=cmap(norm(dbc)), lw=1.3, zorder=3)
        if crosses:
            inside = (xs >= in_lo) & (xs <= in_hi) & (y >= out_lo) & (y <= out_hi)
            if inside.any():
                k = np.where(inside)[0][-1]
                star = " *" if (m, n, s) in sig_cross else ""
                box_labels.append((y[k], xs[k], f"{m}x{n}  -{dbc:.0f}{star}", fail))
        elif dbc <= label_dbc:
            j = np.where(inview)[0][-1]
            ax.annotate(
                f"{m}x{n}  -{dbc:.0f}",
                (xs[j], y[j]),
                fontsize=6.5,
                ha="right",
                va="bottom",
                bbox={"fc": "white", "alpha": 0.6, "lw": 0, "pad": 0.5},
                zorder=7,
            )

    # Labels for products crossing the box: right of the box, spread to avoid overlap.
    box_labels.sort(key=lambda t: t[0])
    gap = 0.04 * (y1 - y0)
    x_lab = min(in_hi + 0.03 * (x1 - x0), x1 - 0.12 * (x1 - x0))
    prev = -np.inf
    for y_pt, x_pt, text, fail in box_labels:
        y_lab = max(y_pt, prev + gap)
        prev = y_lab
        ax.annotate(
            text,
            (x_pt, y_pt),
            xytext=(x_lab, y_lab),
            fontsize=7.5,
            fontweight="bold",
            color="red" if fail else "k",
            ha="left",
            va="center",
            zorder=10,
            bbox={"fc": "white", "alpha": 0.85, "lw": 0.5, "ec": "0.6", "pad": 1.0},
            arrowprops={"arrowstyle": "-", "lw": 0.7, "color": "0.3"},
        )

    yd = np.abs(xs + desired_sign * f_lo)
    ax.plot(xs, yd, color="k", lw=2.8, zorder=8)
    ax.add_patch(Rectangle((in_lo, out_lo), bw_in, bw_out, fill=False, ec="k", lw=1.6, zorder=9))
    if sig_in is not None:
        ax.add_patch(
            Rectangle(
                (sig_in[0], sig_out[0]),
                sig_in[1] - sig_in[0],
                sig_out[1] - sig_out[0],
                fill=False,
                ec="purple",
                lw=2.2,
                ls="--",
                zorder=9,
            ),
        )
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel(f"input frequency ({un})")
    ax.set_ylabel(f"output frequency ({un})")
    ax.grid(alpha=0.25)

    if not cross:
        verdict = "no mixing product crosses the box"
    else:
        w = cross[0]
        verdict = (
            f"{len(cross)} product(s) cross the box; worst {w['m']}x{w['n']} at -{w['dbc']:.0f} dBc"
        )
        if spec_dbc is not None:
            nfail = sum(1 for r in cross if r["dbc"] < spec_dbc)
            verdict += (
                f"; {nfail} fail the -{spec_dbc:g} dBc spec"
                if nfail
                else f"; all meet the -{spec_dbc:g} dBc spec"
            )
    if sig_in is not None:
        sc = sorted(sig_cross.values(), key=lambda r: r["dbc"])
        where = f"in {sum(sig_in) / 2:g} / out {sum(sig_out) / 2:g} {un}"
        if not sc:
            verdict += f"\nsignal box ({signal_bw:g} {un}, {where}): no product crosses"
        else:
            verdict += (
                f"\nsignal box ({signal_bw:g} {un}, {where}): {len(sc)} cross, "
                f"worst {sc[0]['m']}x{sc[0]['n']} at -{sc[0]['dbc']:.0f} dBc"
            )
    ax.set_title(f"Spur chart, LO = {f_lo:g} {un}\n{verdict}", fontsize=11)

    handles = [
        Patch(fc="tab:blue", alpha=0.25, label=f"input band {in_lo:g}-{in_hi:g} {un}"),
        Patch(fc="gold", alpha=0.5, label=f"output band {out_lo:g}-{out_hi:g} {un}"),
        Patch(fc="none", ec="k", label="box (in x out)"),
        Line2D([], [], color="k", lw=2.8, label="desired (1x1)"),
        Line2D([], [], color="gray", lw=2.2, ls="--", label="product in box, meets spec"),
    ]
    if spec_dbc is not None:
        handles.append(Line2D([], [], color="red", lw=3, label="product in box, fails spec"))
    if sig_in is not None:
        handles.append(
            Line2D([], [], color="purple", lw=2.2, ls="--", label="signal box (* = crosses)"),
        )
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=3,
        fontsize=8,
        frameon=False,
    )
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    ax.figure.colorbar(sm, ax=ax, label="product suppression (dB below desired)")
    return ax


def plot_spur_sweep(  # noqa: C901, PLR0912, PLR0915
    sweep: xr.Dataset,
    spec_dbc: float | None = None,
    x: str = "input",
    show_dbc: float | None = None,
    label_dbc: float | None = 70.0,
    y_range: tuple[Any, Any] | None = None,
    title: str | None = None,
    axes: tuple[Axes, Axes] | None = None,
) -> tuple[Axes, Axes]:
    """Swept-input spur view: spur frequencies vs. sweep, with a spur-free strip.

    Top panel: every mixing product as a line against the swept frequency, coloured by its
    suppression (red = strong, green = weak) with the spur's modulation width shaded. The
    gold ribbon is the output band that must stay clean and the thick black line is the
    desired product. Where a spur crosses the ribbon it is overdrawn: bold red if it
    violates ``spec_dbc``, dashed otherwise, and labelled ``mxn -level`` at the crossing.

    Bottom panel: the worst in-band spur suppression at each sweep point against the spec.
    The strip along the bottom marks the spur-free ranges (green) and the violating ranges
    (red), i.e. the frequencies you can actually use.

    Args:
        sweep: Result of :func:`sweep_spurs` (or ``Mixer.spur_sweep``).
        spec_dbc: Required suppression in dB below the desired product (default: the value
            stored in the sweep).
        x: Swept-axis variable: ``"input"`` (mixer input frequency), ``"output"`` (desired
            output frequency) or ``"lo"`` (LO frequency, useful for a swept/tracking LO).
        show_dbc: Hide spurs weaker than this unless they cross the ribbon
            (default: ``spec_dbc + 20`` or 80).
        label_dbc: Label non-crossing spurs at or below this suppression (``None`` = none).
        y_range: ``(low, high)`` limits of the top panel (Hz or Quantity). Default: the output
            band plus a margin of three-quarters of the sweep span (at least 4 output widths).
        title: Optional figure title.
        axes: Optional pair of axes ``(top, bottom)``.

    Returns:
        The ``(top, bottom)`` axes.
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    spec = sweep.attrs["spec_dbc"] if spec_dbc is None else spec_dbc
    spec = None if (spec is None or np.isnan(spec)) else float(spec)
    f_in = sweep["frequency"].to_numpy()
    xvals = {"input": f_in, "output": sweep["out_center"].to_numpy(), "lo": sweep["lo"].to_numpy()}[
        x
    ]
    order = np.argsort(xvals)
    xs = xvals[order]
    out_c = sweep["out_center"].to_numpy()[order]
    obw = float(sweep.attrs["out_bw"])
    sign = int(sweep.attrs["desired_sign"])
    bw = float(sweep.attrs["signal_bw"])
    floor = float(sweep.attrs["floor_dbc"])

    if y_range is not None:
        y_bot, y_top = (float(to_hz(v)) for v in y_range)
    else:
        pad = max(0.75 * float(out_c.max() - out_c.min()), 4 * obw, 1e-9 * out_c.max())
        y_bot, y_top = max(0.0, float(out_c.min()) - pad), float(out_c.max()) + pad
    div, un = freq_unit(np.concatenate([xs, [y_top]]))
    y_bot, y_top = y_bot / div, y_top / div

    if show_dbc is None:
        show_dbc = (spec + 20) if spec is not None else 80.0
    if axes is None:
        _, axes = plt.subplots(
            2,
            1,
            figsize=(11, 8.5),
            sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1]},
            constrained_layout=True,
        )
    ax1, ax2 = axes

    norm = colors.Normalize(vmin=0, vmax=show_dbc)
    cmap = cm.RdYlGn
    dbc = sweep["dbc"].to_numpy()
    inb = sweep["in_band"].to_numpy()[order]
    lvl = sweep["level"].to_numpy()[order]
    fc = sweep["f_center"].to_numpy()[order]
    flo = sweep["f_low"].to_numpy()[order]
    fhi = sweep["f_high"].to_numpy()[order]
    labels = sweep["label"].to_numpy()
    m_arr, n_arr = sweep["m"].to_numpy(), sweep["n"].to_numpy()
    xd = xs / div

    # Output band ribbon (the part that must stay clean)
    ax1.fill_between(
        xd,
        (out_c - obw / 2) / div,
        (out_c + obw / 2) / div,
        color="gold",
        alpha=0.45,
        lw=0,
        zorder=1,
    )
    ax1.plot(xd, out_c / div, color="k", lw=2.4, zorder=8)

    pending = []  # labels placed after the loop
    for k in range(dbc.size):
        crosses = bool(inb[:, k].any())
        if dbc[k] > show_dbc and not crosses:
            continue
        y = fc[:, k] / div
        if not np.any((y >= y_bot) & (y <= y_top)):
            continue
        color = cmap(norm(dbc[k]))
        if bw > 0 and m_arr[k] > 0:
            ax1.fill_between(xd, flo[:, k] / div, fhi[:, k] / div, color=color, alpha=0.18, lw=0)
        ax1.plot(xd, y, color=color, lw=1.2, zorder=3)
        if crosses:
            bad = inb[:, k] & (lvl[:, k] < spec) if spec is not None else np.zeros_like(inb[:, k])
            fails = bool(bad.any())
            ax1.plot(
                xd, np.where(inb[:, k] & ~bad, y, np.nan), color="0.25", lw=2.2, ls="--", zorder=7
            )
            if fails:
                ax1.plot(xd, np.where(bad, y, np.nan), color="red", lw=3.2, zorder=7)
            j = int(np.where(bad if fails else inb[:, k])[0][-1])
            pending.append((y[j], xd[j], f"{m_arr[k]}x{n_arr[k]}  -{dbc[k]:.0f}", fails))
        elif label_dbc is not None and dbc[k] <= label_dbc:
            vis = np.where((y >= y_bot) & (y <= y_top))[0]
            j = int(vis[-1])
            ax1.annotate(
                f"{m_arr[k]}x{n_arr[k]}  -{dbc[k]:.0f}",
                (xd[j], y[j]),
                fontsize=6.5,
                ha="right",
                va="bottom",
                bbox={"fc": "white", "alpha": 0.6, "lw": 0, "pad": 0.5},
                zorder=6,
                annotation_clip=True,
            )
    for y_pt, x_pt, text, fail in pending:
        ax1.annotate(
            text,
            (x_pt, y_pt),
            xytext=(6, 10),
            textcoords="offset points",
            fontsize=7.5,
            fontweight="bold",
            color="red" if fail else "k",
            bbox={"fc": "white", "alpha": 0.85, "lw": 0.5, "ec": "0.6", "pad": 1.0},
            arrowprops={"arrowstyle": "-", "lw": 0.7, "color": "0.3"},
            zorder=10,
            annotation_clip=True,
        )
    ax1.set_ylim(y_bot, y_top)
    ax1.set_xlim(xd[0], xd[-1])
    ax1.set_ylabel(f"spur / output frequency ({un})")
    ax1.grid(alpha=0.25)
    handles = [
        Line2D([], [], color="k", lw=2.4, label="desired (1x1)"),
        Patch(fc="gold", alpha=0.45, label="output band (must stay clean)"),
        Line2D([], [], color="0.25", lw=2.2, ls="--", label="spur in band, meets spec"),
    ]
    if spec is not None:
        handles.append(
            Line2D([], [], color="red", lw=3.2, label=f"spur in band, fails -{spec:g} dBc")
        )
    ax1.legend(handles=handles, loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
    sb = "IN+LO" if sign > 0 else "|IN-LO|"
    ax1.set_title(
        title
        or (
            f"Spur sweep: output = {sb}, signal {fmt_freq(bw)}, "
            f"output band {fmt_freq(obw)}  [{sweep.attrs['table']}]"
        ),
        fontsize=11,
    )

    # Bottom: worst in-band suppression + spur-free strip
    worst = sweep["worst_dbc"].to_numpy()[order]
    shown = np.where(np.isnan(worst), floor, worst)
    ax2.plot(xd, shown, color="tab:blue", lw=1.6, zorder=3)
    ymax = floor + 8
    ax2.set_ylim(-11, ymax)
    if spec is not None:
        ax2.axhline(spec, color="k", ls="--", lw=1)
        ax2.text(xd[0], spec, f" spec -{spec:g} dBc", va="bottom", fontsize=8)
        ok = np.isnan(worst) | (worst >= spec)
        ax2.fill_between(xd, spec, shown, where=shown < spec, color="red", alpha=0.35, lw=0)
    else:
        ok = np.isnan(worst)
    ax2.fill_between(xd, -9, -2, where=ok, step="mid", color="tab:green", lw=0)
    ax2.fill_between(xd, -9, -2, where=~ok, step="mid", color="tab:red", lw=0)
    ax2.set_ylabel("worst in-band spur\n(dB below desired)")
    ax2.set_xlabel(
        {"input": "mixer input", "output": "mixer output", "lo": "LO"}[x] + f" frequency ({un})",
    )
    ax2.grid(alpha=0.25)
    ax2.legend(
        handles=[
            Patch(
                fc="tab:green",
                label="spur-free / meets spec" if spec is not None else "no spur in band",
            ),
            *([Patch(fc="tab:red", label="spur violates spec")] if spec is not None else []),
        ],
        loc="lower right",
        fontsize=8,
        framealpha=0.9,
    )
    return ax1, ax2
