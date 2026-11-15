"""Frequency planning and translation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr
from pint import Quantity

from xcvr.spurs import Spur, SpurTable, compute_spurs, sweep_spurs, to_hz


@dataclass
class FrequencyPlan:
    """Tracks the RF cascade axis and the local carrier frequency separately."""

    rf: np.ndarray  # shared cascade axis, never changes across the chain
    carrier: np.ndarray  # local operating frequency for this device/band

    @classmethod
    def passthrough(cls, frequency: np.ndarray) -> FrequencyPlan:
        """Default plan for a device not yet assigned to a system — RF and carrier are the same."""
        return cls(rf=frequency, carrier=frequency)

    @property
    def rf_da(self) -> xr.DataArray:
        """RF, index, and input frequency and does not change through the chain."""
        return xr.DataArray(self.rf, dims=("frequency",), coords={"frequency": self.rf})

    @property
    def is_translated(self) -> bool:
        """Whether the frequency has been translated - RF and carrier will not match."""
        return not np.array_equal(self.rf, self.carrier)

    @property
    def carrier_da(self) -> xr.DataArray:
        """Carrier frequency which is the local operating frequency of each device."""
        return xr.DataArray(self.carrier, dims=("frequency",), coords={"frequency": self.carrier})

    def translate(self, lo_freq: float, sideband: str = "high") -> FrequencyPlan:
        """Translate the frequency plan given the LO frequency and sideband."""
        if sideband == "high":
            next_carrier = self.carrier + lo_freq
        else:
            next_carrier = np.abs(self.carrier - lo_freq)
        return FrequencyPlan(rf=self.rf, carrier=next_carrier)

    def label(self, da: xr.DataArray) -> xr.DataArray:
        """
        Relabel a DataArray's frequency axis to the RF grid and attach
        carrier_freq as a coord. This is the only place that logic lives.
        """
        da = da.assign_coords(frequency=("frequency", self.rf))
        da = da.assign_coords(carrier_freq=("frequency", self.carrier))
        da.coords["frequency"].attrs["long_name"] = "RF Index Frequency"
        da.coords["carrier_freq"].attrs["long_name"] = "Carrier Frequency"
        return da


class MixerMixin:
    """Adds frequency translation (and optional spur analysis) to any Device subclass."""

    def __init__(
        self,
        *args,
        lo_freq: Quantity,
        sideband: str = "low",
        spur_table: SpurTable | None = None,
        spur_ref_power: Quantity | float | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lo_freq = lo_freq
        self.sideband = sideband
        # Datasheet m x n spur table. Optional: only needed for spur analysis.
        self.spur_table = spur_table
        # Input power (dBm) the spur table was measured at. If set, tone tracing moves
        # products with m > 1 by (m - 1) dB per dB of drive change.
        self.spur_ref_power = spur_ref_power

        # Ordered (lo_freq, sideband) stages this device represents.
        # A plain single mixer is just its own one stage; a collapsed
        # System overrides this with the full chain it stood in for.
        self.stages: list[tuple[Quantity, str]] = [(lo_freq, sideband)]

    def apply_translation(self, plan: FrequencyPlan) -> FrequencyPlan:
        """Apply this device's translation stage(s), in order, to a plan."""
        for lo, sideband in self.stages:
            lof = lo.to("Hz").magnitude if isinstance(lo, Quantity) else lo
            plan = plan.translate(lof, sideband)
        return plan

    # ------------------------------------------------------------------ spurs
    @property
    def desired_sign(self) -> int:
        """+1 if the desired product is ``IN + LO`` (sideband "high"), -1 for ``|IN - LO|``."""
        return 1 if self.sideband == "high" else -1

    def _spur_setup(self) -> tuple[SpurTable, float]:
        """Validated (table, LO in Hz) for spur analysis."""
        if len(self.stages) != 1:
            raise ValueError(
                f"{self.name!r} stands in for {len(self.stages)} mixer stages; "
                "expand the sub-system to analyse each mixer's spurs.",
            )
        if self.spur_table is None:
            raise ValueError(
                f"{self.name!r} has no spur_table. Pass spur_table=SpurTable(...) "
                "(SpurTable.default() is illustrative only).",
            )
        return self.spur_table, float(to_hz(self.lo_freq))

    def spurs(
        self,
        in_freq: Quantity | float,
        signal_bw: Quantity | float = 0.0,
        out_bw: Quantity | float | None = None,
        max_order: int | None = None,
        rf_backoff_db: float = 0.0,
    ) -> list[Spur]:
        """All mixing products for one input frequency, worst first (see ``compute_spurs``).

        Args:
            in_freq: Mixer input centre frequency (Hz if a bare number).
            signal_bw: Input signal bandwidth (0 = CW).
            out_bw: Output band to protect (default: ``signal_bw``).
            max_order: Ignore products with m + n above this.
            rf_backoff_db: Input drive below the spur table's reference level.
        """
        table, lo = self._spur_setup()
        return compute_spurs(
            in_freq,
            lo,
            table,
            max_order=max_order,
            rf_backoff_db=rf_backoff_db,
            desired_sign=self.desired_sign,
            out_bw=signal_bw if out_bw is None else out_bw,
            in_bw=signal_bw,
        )

    def spur_sweep(
        self,
        in_freq: xr.DataArray | np.ndarray | Quantity,
        signal_bw: Quantity | float,
        out_bw: Quantity | float | None = None,
        max_order: int | None = None,
        rf_backoff_db: float = 0.0,
        spec_dbc: float | None = None,
        npts: int = 1001,
        use_overlap_fraction: bool = False,
    ) -> xr.Dataset:
        """Spurs while the input is swept across ``in_freq`` (see ``sweep_spurs``).

        The sweep is resampled to at least ``npts`` evenly spaced points so a spur that
        crosses the output band between two coarse points is not missed. Plot it with
        :func:`xcvr.spurs.plot_spur_sweep`.

        Args:
            in_freq: Mixer input frequencies to sweep (only min and max matter if resampled).
            signal_bw: Instantaneous signal bandwidth.
            out_bw: Output band to protect (default: ``signal_bw``).
            max_order: Ignore products with m + n above this.
            rf_backoff_db: Input drive below the spur table's reference level.
            spec_dbc: Required spur suppression, stored in the result for plotting.
            npts: Minimum number of sweep points.
            use_overlap_fraction: Judge in-band spurs by the power that lands in the band
                instead of their full table level (default, conservative).
        """
        table, lo = self._spur_setup()
        f = np.asarray(to_hz(getattr(in_freq, "data", in_freq)), dtype=float).ravel()
        if f.size < npts:
            f = np.linspace(f.min(), f.max(), npts)
        return sweep_spurs(
            table,
            np.sort(f),
            lo,
            desired_sign=self.desired_sign,
            signal_bw=signal_bw,
            out_bw=out_bw,
            max_order=max_order,
            rf_backoff_db=rf_backoff_db,
            spec_dbc=spec_dbc,
            use_overlap_fraction=use_overlap_fraction,
        )
