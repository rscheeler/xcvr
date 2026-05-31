"""Frequency planning and translation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr


@dataclass
class FrequencyPlan:
    """Tracks the RF cascade axis and the local carrier frequency separately."""

    rf: np.ndarray  # shared cascade axis — never changes across the chain
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
    """Adds frequency translation to any Device subclass."""

    def __init__(self, *args, lo_freq: Quantity, sideband: str = "low", **kwargs):
        super().__init__(*args, **kwargs)
        self.lo_freq = lo_freq
        self.sideband = sideband
