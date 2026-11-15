"""
Tone tracing: absolute-power spur spectrum at any node of a :class:`xcvr.System`.

A swept CW source goes in; every mixing product is followed through the chain. Each device
applies its own S21 at the tone's *actual* frequency (so filters reject spurs, amplifiers
amplify them) and each mixer with a ``spur_table`` splits every incoming tone into its
``m x n`` products. The result is the "spectrum sweep at node k" view: output frequency vs.
power (dBm) with one trace per mixing path, and the path name for every trace.

Model and limits
----------------
* Linear tone tracing: no compression, no amplifier harmonics, no noise. Spurs are created
  only by mixers that carry a ``spur_table``.
* Power of an ``m x n`` product = (tone power + conversion gain at the tone's input
  frequency) - table dBc, where the table is referenced to the desired product. If the
  mixer has ``spur_ref_power`` (the input power the table was measured at), products with
  ``m > 1`` also move ``(m - 1)`` dB per dB of drive change.
* LO harmonics (``m = 0`` products) are generated once per mixer, from its strongest
  incoming tone.
* Device gain is read from each device's native S-parameters (held constant outside the
  data range), *not* from the cascade's interpolated grid, so out-of-band spurs see the real
  filter response. Nested systems are flattened.
* Frequency lookup matches xcvr's cascade: plain devices are evaluated at the tone's actual
  frequency, mixers at the frequency its ancestor entered the system (xcvr indexes a mixer's
  S-parameters by the system RF grid). For the desired signal this reproduces
  ``System.gain`` exactly; spurs inherit the lookup frequency of their source tone.
* Mixers without a ``spur_table`` pass only their desired product.

Checks reported alongside the trace (so a result is not trusted blindly)
------------------------------------------------------------------------
* **Extrapolation**: a tone whose frequency lies outside a device's S-parameter range is
  flagged (``extrapolated`` in the dataset, hollow markers on the plot); its gain there is the
  edge value, not data.
* **Pruning**: ``SpurTrace.pruning`` counts what ``max_order``, ``min_power_dbm`` and
  ``max_tones`` removed at each device and the strongest removed level.
* **Compression**: ``SpurTrace.compression`` compares the total power at each device output
  with its output P1dB. Compression is *flagged, not modelled*.
* **Noise floor**: the desired path's cascaded noise (Friis, kT0 at the input) at every node, so
  spurs can be judged against the noise in the channel. Drawn by :func:`plot_spectrum_sweep`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr
from pint import Quantity

from xcvr.spurs import _products, fmt_freq, freq_unit, to_hz

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = ["SpurTrace", "Tone", "ToneSource", "plot_spectrum_sweep", "trace_tones"]


def _dbm(value: Any) -> Any:
    if isinstance(value, Quantity):
        return value.to("dBm").magnitude
    return value


@dataclass
class ToneSource:
    """A CW source at the system input.

    Args:
        freq: Frequency (Hz or Quantity). An array makes this a swept source.
        power: Power in dBm (Quantity or number); scalar or an array matching ``freq``.
        name: Label used in path names.
    """

    freq: Any
    power: Any = 0.0
    name: str = "RF"


@dataclass
class Tone:
    """One trace: a single mixing path followed over the whole sweep."""

    freq: np.ndarray  # Hz, per sweep point
    power: np.ndarray  # dBm, per sweep point
    source: str
    steps: tuple[str, ...] = ()
    desired: bool = True
    order: int = 0
    # Frequency (Hz, per sweep point) at which this tone's ancestor entered the system. xcvr
    # indexes every mixer's S-parameters by this system-input frequency, so tracing does too.
    origin: np.ndarray | None = None
    # True where any device so far had to hold its edge value (tone outside its S-parameters).
    extrap: np.ndarray | None = None
    extrap_devs: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        """Path name, e.g. ``RF > MX1:2x3-``."""
        return " > ".join((self.source, *self.steps))


@dataclass
class SpurTrace:
    """Result of :func:`trace_tones`. Node ``k`` is the output of the k-th device (1-based);
    node 0 is the system input."""

    sweep: np.ndarray
    sweep_name: str
    node_names: list[str]
    node_tones: list[list[Tone]] = field(repr=False)
    # Desired-path frequency (Hz) and cascaded noise density (dBm/Hz) per node and sweep point.
    node_noise: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list, repr=False)
    _pruning: list[dict] = field(default_factory=list, repr=False)
    _compression: list[dict] = field(default_factory=list, repr=False)

    @property
    def pruning(self) -> Any:
        """DataFrame: per device, tones kept and dropped by ``max_order`` / the power floor /
        ``max_tones``, the strongest dropped level (dBm) and whether it matters (``significant``:
        within 70 dB of the node's strongest trace)."""
        import pandas as pd

        return pd.DataFrame(self._pruning)

    @property
    def compression(self) -> Any:
        """DataFrame: per device, the highest total output power over the sweep, its output
        P1dB at that point, the margin (dB, negative = above P1dB) and ``flagged``."""
        import pandas as pd

        return pd.DataFrame(self._compression)

    def _index(self, node: int | str) -> int:
        if isinstance(node, str):
            hits = [i for i, n in enumerate(self.node_names) if n == node]
            if not hits:
                raise KeyError(f"No node named {node!r}; nodes: {self.node_names}")
            if len(hits) > 1:
                raise KeyError(f"{node!r} is ambiguous; use the node number {hits}")
            return hits[0]
        return int(node) % len(self.node_names)

    def dataset(self, node: int | str = -1, min_power_dbm: float | None = None) -> xr.Dataset:
        """Tones at a node as a Dataset with dims ``(sweep, tone)``.

        Variables: ``freq`` (Hz) and ``power_dbm`` ``(sweep, tone)``; ``peak_dbm`` per tone;
        coords ``label, source, desired, order`` on ``tone``. Strongest tone first.
        """
        i = self._index(node)
        tones = sorted(self.node_tones[i], key=lambda t: -float(np.max(t.power)))
        if min_power_dbm is not None:
            tones = [t for t in tones if float(np.max(t.power)) >= min_power_dbm]
        n = self.sweep.size
        freq = np.column_stack([t.freq for t in tones]) if tones else np.empty((n, 0))
        power = np.column_stack([t.power for t in tones]) if tones else np.empty((n, 0))
        extrap = (
            np.column_stack([_extrap_of(t, n) for t in tones])
            if tones
            else np.empty((n, 0), dtype=bool)
        )
        data_vars = {
            "freq": (("sweep", "tone"), freq),
            "power_dbm": (("sweep", "tone"), power),
            "peak_dbm": ("tone", power.max(axis=0) if tones else np.empty(0)),
            "extrapolated": (("sweep", "tone"), extrap),
        }
        if self.node_noise:
            nf_, nd_ = self.node_noise[i]
            data_vars["noise_freq"] = ("sweep", nf_)
            data_vars["noise_dbm_hz"] = ("sweep", nd_)
        return xr.Dataset(
            data_vars,
            coords={
                "sweep": self.sweep,
                "tone": np.arange(len(tones)),
                "label": ("tone", [t.label for t in tones]),
                "source": ("tone", [t.source for t in tones]),
                "desired": ("tone", [t.desired for t in tones]),
                "order": ("tone", [t.order for t in tones]),
                "extrap_devices": ("tone", [", ".join(t.extrap_devs) for t in tones]),
            },
            attrs={
                "node": i,
                "node_name": self.node_names[i],
                "sweep_name": self.sweep_name,
                "units": "Hz, dBm",
            },
        )

    def summary(self, node: int | str = -1, n: int = 15) -> Any:
        """DataFrame of the ``n`` strongest traces at a node (frequency span, peak power)."""
        import pandas as pd

        ds = self.dataset(node)
        rows = []
        for k in range(min(n, ds.sizes["tone"])):
            f, p = ds.freq[:, k].to_numpy(), ds.power_dbm[:, k].to_numpy()
            j = int(np.argmax(p))
            rows.append(
                {
                    "path": str(ds.label[k].item()),
                    "desired": bool(ds.desired[k]),
                    "f_min_Hz": f.min(),
                    "f_max_Hz": f.max(),
                    "peak_dBm": p[j],
                    "f_at_peak_Hz": f[j],
                    "sweep_at_peak_Hz": float(self.sweep[j]),
                    "extrapolated_in": str(ds.extrap_devices[k].item()),
                    "extrapolated_pct": 100.0 * float(ds.extrapolated[:, k].mean()),
                },
            )
        return pd.DataFrame(rows)


# ------------------------------------------------------------------ tracing
def _flatten(devices: list[Any]) -> list[Any]:
    """Original (un-interpolated) devices, with nested Systems expanded."""
    out: list[Any] = []
    for d in devices:
        if hasattr(d, "_devices"):
            out += _flatten(d._devices)  # noqa: SLF001
        else:
            out.append(d)
    return out


def _gain_db(dev: Any, f: np.ndarray) -> np.ndarray:
    """S21 (dB) of a device at arbitrary frequencies from its native S-parameters."""
    net = dev.network
    g = np.maximum(np.asarray(net.s_db[:, 1, 0], dtype=float), -300.0)
    return np.interp(f, net.f, g)  # holds the edge values outside the data


def _outside(dev: Any, f: np.ndarray) -> np.ndarray:
    """True where ``f`` is outside the device's S-parameter frequency range."""
    fr = np.asarray(dev.network.f, dtype=float)
    tol = 1e-9 * float(fr.max())
    return (f < fr.min() - tol) | (f > fr.max() + tol)


def _prop_at(da: Any, f: np.ndarray, unit: str) -> np.ndarray:
    """A device property (``nf``, ``p1db``) at frequencies ``f``; non-finite -> ``inf``."""
    vals = np.asarray(da.data.to(unit).magnitude, dtype=float)
    vals = vals.reshape(vals.shape[0], -1)[:, 0] if vals.ndim > 1 else vals
    fr = np.asarray(da["frequency"].to_numpy(), dtype=float)
    ok = np.isfinite(vals)
    if not ok.any():
        return np.full(np.shape(f), np.inf)
    return np.interp(f, fr[ok], vals[ok]) if ok.sum() > 1 else np.full(np.shape(f), vals[ok][0])


def _extrap_of(t: Tone, n: int) -> np.ndarray:
    return np.zeros(n, dtype=bool) if t.extrap is None else t.extrap


def _step_label(m: int, n: int, s: int) -> str:
    return f"{m}x{n}{'+' if s > 0 else '-'}"


def _mix(
    tone: Tone,
    mixer: Any,
    lo: float,
    g: np.ndarray,
    max_order: int | None,
    dropped: list[float],
) -> list[Tone]:
    """All products of one tone through one mixer (``m >= 1``). Products removed by
    ``max_order`` are not built; their peak power goes into ``dropped``."""
    table, sd = mixer.spur_table, mixer.desired_sign
    pdes = tone.power + g
    ref = getattr(mixer, "spur_ref_power", None)
    backoff = 0.0 if ref is None else float(_dbm(ref)) - tone.power
    out = []
    for m, n, s in _products(table, None):
        if m == 0:
            continue
        is_des = m == 1 and n == 1 and s == sd
        dbc = 0.0 if is_des else table.level(m, n) + (m - 1) * backoff
        if max_order is not None and m + n > max_order:
            dropped.append(float(np.max(pdes - dbc)))
            continue
        out.append(
            Tone(
                freq=np.abs(m * tone.freq + s * n * lo),
                power=pdes - dbc,
                source=tone.source,
                steps=(*tone.steps, f"{mixer.name}:{_step_label(m, n, s)}"),
                desired=tone.desired and is_des,
                order=tone.order + m + n,
                origin=tone.origin,
            ),
        )
    return out


def _translate(tone: Tone, mixer: Any, g: np.ndarray) -> Tone:
    """Desired product only (mixer without a spur table), through every stage."""
    return Tone(
        _stage_freq(tone.freq, mixer),
        tone.power + g,
        tone.source,
        (*tone.steps, f"{mixer.name}:1x1"),
        tone.desired,
        tone.order + 2,
        tone.origin,
    )


def _stage_freq(f: np.ndarray, mixer: Any) -> np.ndarray:
    for lo, sideband in mixer.stages:
        f = f + float(to_hz(lo)) if sideband == "high" else np.abs(f - float(to_hz(lo)))
    return f


def _mark(new: list[Tone], old: Tone, dev: Any, out: np.ndarray) -> list[Tone]:
    """Carry the extrapolation flags of ``old`` (plus this device's) onto ``new``."""
    ex = _extrap_of(old, out.size) | out
    devs = (*old.extrap_devs, dev.name) if out.any() else old.extrap_devs
    for t in new:
        t.extrap, t.extrap_devs = ex, devs
    return new


def trace_tones(  # noqa: C901, PLR0912, PLR0915
    system: Any,
    sources: ToneSource | list[ToneSource],
    max_order: int | None = 7,
    min_power_dbm: float = -150.0,
    max_tones: int = 1500,
    compression_margin_db: float = 0.0,
    t_source: float = 290.0,
    warn: bool = True,
) -> SpurTrace:
    """Follow every mixing product of the source tone(s) through the system.

    Args:
        system: A :class:`xcvr.System` (nested systems are flattened).
        sources: One or more :class:`ToneSource`. Arrays must share one length; scalar
            values are broadcast. The first swept source defines the sweep axis.
        max_order: Ignore products with ``m + n`` above this in every mixer.
        min_power_dbm: Drop a trace once its peak power falls below this.
        max_tones: Keep at most this many (strongest) traces after each device.
        compression_margin_db: Flag a device whose total output power is within this many dB
            of its output P1dB (0 = flag only at or above P1dB).
        t_source: Source noise temperature (K) for the noise floor.
        warn: Emit a :class:`UserWarning` summarising extrapolation, significant pruning and
            compression. The same facts are always in the returned object.

    Returns:
        :class:`SpurTrace`; call ``.dataset(node)`` or ``.summary(node)``, or plot with
        :func:`plot_spectrum_sweep`. ``.pruning`` and ``.compression`` hold the checks.
    """
    from xcvr.frequency import MixerMixin

    sources = [sources] if isinstance(sources, ToneSource) else list(sources)
    arrays = [np.atleast_1d(np.asarray(to_hz(s.freq), dtype=float)) for s in sources]
    sizes = {a.size for a in arrays if a.size > 1}
    if len(sizes) > 1:
        raise ValueError("Swept sources must all have the same number of points.")
    npts = sizes.pop() if sizes else 1
    swept = next((i for i, a in enumerate(arrays) if a.size > 1), 0)
    sweep = np.broadcast_to(arrays[swept], (npts,)).copy()

    tones = []
    for s, a in zip(sources, arrays, strict=True):
        p = np.broadcast_to(np.asarray(_dbm(s.power), dtype=float), (npts,)).astype(float)
        fa = np.broadcast_to(a, (npts,)).astype(float)
        tones.append(Tone(fa, p, s.name, origin=fa, extrap=np.zeros(npts, dtype=bool)))

    devs = _flatten(list(system._devices))  # noqa: SLF001
    names = ["input"]
    snaps = [list(tones)]
    k_t0 = 10.0 * np.log10(1.380649e-23 * t_source / 1e-3)  # dBm/Hz
    # desired path, tracked on its own so the noise floor survives pruning
    d_freq, d_origin = sweep.copy(), sweep.copy()
    f_lin, g_lin = np.ones(npts), np.ones(npts)
    noise = [(d_freq.copy(), np.full(npts, k_t0))]
    pruning: list[dict] = []
    compression: list[dict] = []

    for dev in devs:
        mixing = isinstance(dev, MixerMixin)
        table = getattr(dev, "spur_table", None) if mixing else None
        nxt: list[Tone] = []
        dropped_order: list[float] = []
        if mixing and table is not None:
            if len(dev.stages) != 1:
                raise ValueError(f"{dev.name!r} stands in for several mixers; expand it.")
            lo = float(to_hz(dev.lo_freq))
            for t in tones:
                out = _outside(dev, t.origin)
                nxt += _mark(
                    _mix(t, dev, lo, _gain_db(dev, t.origin), max_order, dropped_order),
                    t,
                    dev,
                    out,
                )
            # LO harmonics (m = 0): once per mixer, from its strongest incoming tone
            ref = max(tones, key=lambda t: float(np.max(t.power)), default=None)
            if ref is not None:
                pdes = ref.power + _gain_db(dev, ref.origin)
                lo_tones = []
                for m, n, s in _products(table, None):
                    if m != 0:
                        continue
                    if max_order is not None and n > max_order:
                        dropped_order.append(float(np.max(pdes - table.level(0, n))))
                        continue
                    lo_tones.append(
                        Tone(
                            freq=np.full(npts, n * lo),
                            power=pdes - table.level(0, n),
                            source="LO",
                            steps=(f"{dev.name}:0x{n}",),
                            desired=False,
                            order=n,
                            origin=ref.origin,
                        ),
                    )
                nxt += _mark(lo_tones, ref, dev, _outside(dev, ref.origin))
        elif mixing:
            nxt = [
                _mark(
                    [_translate(t, dev, _gain_db(dev, t.origin))], t, dev, _outside(dev, t.origin)
                )[0]
                for t in tones
            ]
        else:
            nxt = [
                _mark(
                    [
                        Tone(
                            t.freq,
                            t.power + _gain_db(dev, t.freq),
                            t.source,
                            t.steps,
                            t.desired,
                            t.order,
                            t.origin,
                        ),
                    ],
                    t,
                    dev,
                    _outside(dev, t.freq),
                )[0]
                for t in tones
            ]

        # desired-path noise (Friis) at the same lookup frequencies the tones use
        f_look = d_origin if mixing else d_freq
        g_db = _gain_db(dev, f_look)
        nf_db = _prop_at(dev.nf, f_look, "dB")
        f_lin = f_lin + (10 ** (nf_db / 10.0) - 1.0) / g_lin
        g_lin = g_lin * 10 ** (g_db / 10.0)
        if mixing:
            d_freq = _stage_freq(d_freq, dev)
        noise.append((d_freq.copy(), k_t0 + 10 * np.log10(f_lin) + 10 * np.log10(g_lin)))

        # compression check: total power at the output vs. output P1dB
        if nxt:
            pw = np.vstack([t.power for t in nxt])
            total = 10 * np.log10(np.sum(10 ** (pw / 10.0), axis=0))
            lead = np.argmax(pw, axis=0)
            look = np.vstack([t.origin if mixing else t.freq for t in nxt])[lead, np.arange(npts)]
            p1 = _prop_at(dev.p1db, look, "dBm")
            margin = p1 - total
            j = int(np.argmin(margin))
            compression.append(
                {
                    "node": len(names),
                    "device": dev.name,
                    "max_out_dbm": float(total[j]),
                    "p1db_dbm": float(p1[j]),
                    "margin_db": float(margin[j]) if np.isfinite(margin[j]) else np.inf,
                    "sweep_hz": float(sweep[j]),
                    "flagged": bool(margin[j] < compression_margin_db),
                },
            )

        # pruning
        n_floor = sum(float(np.max(t.power)) < min_power_dbm for t in nxt)
        nxt = [t for t in nxt if float(np.max(t.power)) >= min_power_dbm]
        nxt.sort(key=lambda t: -float(np.max(t.power)))
        cut = nxt[max_tones:]
        tones = nxt[:max_tones]
        lost = [float(np.max(t.power)) for t in cut] + dropped_order
        top = float(max((np.max(t.power) for t in tones), default=-np.inf))
        worst = max(lost, default=-np.inf)
        pruning.append(
            {
                "node": len(names),
                "device": dev.name,
                "kept": len(tones),
                "dropped_order": len(dropped_order),
                "dropped_floor": n_floor,
                "dropped_cap": len(cut),
                "strongest_dropped_dbm": worst,
                "significant": bool(np.isfinite(worst) and worst > top - 70.0),
            },
        )
        names.append(dev.name)
        snaps.append(list(tones))

    trace = SpurTrace(sweep, sources[swept].name, names, snaps, noise, pruning, compression)
    if warn:
        _warn_checks(trace)
    return trace


def _warn_checks(trace: SpurTrace) -> None:
    msgs = []
    final = trace.node_tones[-1]
    if final:
        top = max(float(np.max(t.power)) for t in final)
        sig = [t for t in final if float(np.max(t.power)) >= top - 70.0]
        bad = [t for t in sig if t.extrap is not None and t.extrap.any()]
        if bad:
            devs = sorted({d for t in bad for d in t.extrap_devs})
            msgs.append(
                f"{len(bad)} of {len(sig)} significant output traces pass a device outside its "
                f"S-parameter range (gain held at the edge value): {', '.join(devs)}."
            )
    for r in trace._pruning:  # noqa: SLF001
        if r["significant"]:
            msgs.append(
                f"{r['device']}: pruning removed a trace at {r['strongest_dropped_dbm']:.1f} dBm "
                "(raise max_order / max_tones / lower min_power_dbm)."
            )
    for r in trace._compression:  # noqa: SLF001
        if r["flagged"]:
            msgs.append(
                f"{r['device']}: output {r['max_out_dbm']:.1f} dBm vs P1dB {r['p1db_dbm']:.1f} dBm "
                f"at {fmt_freq(r['sweep_hz'])} (compression is not modelled)."
            )
    if msgs:
        warnings.warn("trace_tones: " + " | ".join(msgs), UserWarning, stacklevel=3)


# --------------------------------------------------------------------- plot
def plot_spectrum_sweep(  # noqa: C901, PLR0912, PLR0915
    ds: xr.Dataset,
    floor_dbm: float | None = None,
    y_range: tuple[float, float] | None = None,
    x_max: float | Quantity | None = None,
    n_markers: int = 5,
    hover: bool = True,
    title: str | None = None,
    ax: Axes | None = None,
    show_noise: bool = True,
    rbw: float | Quantity = 1e6,
) -> Axes:
    """Spectrum sweep at a node: output frequency vs. power, one trace per mixing path.

    Each arch is one path as the source sweeps; its shape is the chain's gain and filtering
    at that spur's frequency. The desired signal is drawn in black. The ``n_markers``
    strongest traces are numbered, with a list at the right (frequency, power, path and the
    sweep frequency at the peak). With ``hover=True`` and an interactive backend, moving the
    mouse over a point shows the same details for that point.

    Args:
        ds: ``SpurTrace.dataset(node)``.
        floor_dbm: Hide traces whose peak is below this (default: 100 dB below the peak).
        y_range: ``(low, high)`` y limits in dBm.
        x_max: Upper x limit (Hz or Quantity). Default: highest frequency among traces
            within 70 dB of the strongest.
        n_markers: Number of numbered markers / list entries.
        hover: Attach the mouse-over tooltip.
        title: Plot title (default: ``Spectrum Sweep at node k (name)``).
        ax: Axes to draw into.
        show_noise: Draw the desired path's cascaded noise floor (dashed grey) when the
            dataset has it. Spurs below it are buried in the channel's own noise.
        rbw: Resolution bandwidth (Hz or Quantity) the noise floor is integrated over.

    Hollow markers are points where a device's S-parameters did not cover the tone's frequency
    (its gain there is the edge value, not data); the side list and tooltip name the devices.

    Returns:
        The axes. ``ax.figure._spur_hover(event)`` is the tooltip handler (for testing).
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 6.5))
        fig.subplots_adjust(right=0.68, left=0.08, bottom=0.11)
    fig = ax.figure
    if ds.sizes["tone"] == 0:
        ax.set_title("No tones above the power floor")
        return ax
    peak = ds["peak_dbm"].to_numpy()
    top = float(peak.max())
    floor = top - 100.0 if floor_dbm is None else floor_dbm
    keep = [k for k in range(ds.sizes["tone"]) if peak[k] >= floor]
    freq, power = ds["freq"].to_numpy(), ds["power_dbm"].to_numpy()
    sweep = ds["sweep"].to_numpy()
    if x_max is not None:
        xlim = float(to_hz(x_max))
    else:  # the significant traces (within 70 dB of the peak) set the range
        sig = [k for k in keep if peak[k] >= top - 70.0] or keep
        xlim = float(freq[:, sig].max()) * 1.03
    div, un = freq_unit(xlim)
    cmap = plt.get_cmap("tab20")

    ex = ds["extrapolated"].to_numpy() if "extrapolated" in ds else None
    edev = ds["extrap_devices"].to_numpy() if "extrap_devices" in ds.coords else None

    def _flag(k: int, j: int) -> str:
        if ex is None or not ex[j, k]:
            return ""
        return f"\n    outside data: {edev[k]}"

    pts_x, pts_y, meta = [], [], []
    for rank, k in enumerate(keep):
        des = bool(ds["desired"][k])
        x, y = freq[:, k] / div, power[:, k]
        color = "k" if des else cmap(rank % 20)
        ax.plot(
            x, y, "-", color=color, lw=7 if des else 5, alpha=0.35, solid_capstyle="round", zorder=2
        )
        hit = ex[:, k] if ex is not None else np.zeros(x.size, dtype=bool)
        ax.plot(x[~hit], y[~hit], ".", color=color, ms=4, zorder=3)
        if hit.any():  # outside a device's data: hollow, so a guess looks like a guess
            ax.plot(x[hit], y[hit], "o", mfc="none", mec=color, ms=3, mew=0.8, zorder=3)
        pts_x.append(x)
        pts_y.append(y)
        meta += [(k, j) for j in range(x.size)]
    ax.set_xlim(0, xlim / div)
    ax.set_ylim(*(y_range or (floor - 4, top + 8)))
    ax.set_xlabel(f"Frequency ({un})")
    ax.set_ylabel("Power (dBm)")
    ax.grid(alpha=0.5)
    node = int(ds.attrs["node"])
    ax.set_title(
        title or f"Spectrum Sweep at node {node} ({ds.attrs['node_name']})", fontweight="bold"
    )

    if show_noise and "noise_dbm_hz" in ds:
        rb = float(to_hz(rbw))
        nx = ds["noise_freq"].to_numpy() / div
        ny = ds["noise_dbm_hz"].to_numpy() + 10.0 * np.log10(rb)
        use = np.ones(nx.size, dtype=bool)
        if ex is not None:  # draw it where the desired path rests on data
            des = [k for k in range(ds.sizes["tone"]) if bool(ds["desired"][k])]
            if des and (~ex[:, des[0]]).any():
                use = ~ex[:, des[0]]
        order = np.argsort(nx[use])
        ax.plot(nx[use][order], ny[use][order], "--", color="0.35", lw=1.5, zorder=1)
        ax.text(
            float(nx[use].max()),
            float(np.median(ny[use])),
            f"  noise floor ({fmt_freq(rb)} RBW)",
            fontsize=8,
            color="0.35",
            ha="left",
            va="center",
        )

    # numbered markers + side list (strongest first)
    lines = []
    for num, k in enumerate(keep[:n_markers], start=1):
        j = int(np.argmax(power[:, k]))
        fx, py = freq[j, k], power[j, k]
        ax.annotate(
            f"{num}",
            (fx / div, py),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            fontweight="bold",
            zorder=6,
            arrowprops={"arrowstyle": "-", "lw": 0.8},
        )
        lines.append(
            f"{num}) {fmt_freq(fx)}\n    {py:.2f} dBm\n    {ds['label'][k].item()}\n"
            f"    {ds.attrs['sweep_name']} = {fmt_freq(float(sweep[j]))}{_flag(k, j)}",
        )
    if lines:
        ax.text(
            1.02,
            1.0,
            "\n\n".join(lines),
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            family="monospace",
        )

    if hover:
        allx, ally = np.concatenate(pts_x), np.concatenate(pts_y)
        ann = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(14, 14),
            textcoords="offset points",
            bbox={"boxstyle": "round", "fc": "lightyellow", "ec": "0.4"},
            arrowprops={"arrowstyle": "-"},
            fontsize=8,
            zorder=10,
        )
        ann.set_visible(False)

        def on_move(event: Any) -> None:
            if event.inaxes is not ax:
                if ann.get_visible():
                    ann.set_visible(False)
                    fig.canvas.draw_idle()
                return
            disp = ax.transData.transform(np.column_stack([allx, ally]))
            d = np.hypot(disp[:, 0] - event.x, disp[:, 1] - event.y)
            i = int(np.argmin(d))
            if d[i] > 12:  # noqa: PLR2004
                if ann.get_visible():
                    ann.set_visible(False)
                    fig.canvas.draw_idle()
                return
            k, j = meta[i]
            ann.xy = (allx[i], ally[i])
            ann.set_text(
                f"{ds['label'][k].item()}\n{fmt_freq(freq[j, k])}   {power[j, k]:.2f} dBm\n"
                f"{ds.attrs['sweep_name']} = {fmt_freq(float(sweep[j]))}{_flag(k, j)}",
            )
            ann.set_visible(True)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", on_move)
        fig._spur_hover = on_move  # noqa: SLF001
    return ax
