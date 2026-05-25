"""
Convenience model for standard devices.
Utilizes skrf objects to create the skrf.Network objects.
"""

from typing import Any

import schemdraw
import xarray as xr
from pint import Quantity
from skrf import Frequency, Network
from skrf.media import Coaxial, DefinedGammaZ0
from xrench.units import ureg

from xcvr.frequency import MixerMixin
from xcvr.symbols import Symbol
from xcvr.xcvr import Device


class Constant(Device):
    """
    Convenience device for defining as constant values useful
    for initial budgets using datasheet values.
    """

    def __init__(
        self,
        name: str,
        manufacturer: str,
        pn: str,
        frequency: xr.DataArray,
        gain: Quantity,
        noise_gain: Quantity | None = None,
        z0: float = 50.0,
        symbol: Symbol | None = None,
        vsup: Quantity = 0 * ureg.volt,
        isup: Quantity = 0 * ureg.mA,
        **kwargs: Any,
    ) -> None:
        # Generate skrf Frequency object
        frequency = frequency.copy()
        frequency.data = frequency.data.to("Hz")
        freq = Frequency(frequency.values[0], frequency.values[-1], frequency.values.size, "hz")

        # Create media and then attenuator from media
        dgz = DefinedGammaZ0(freq, z0=z0)
        net = dgz.attenuator(gain.to("dB").magnitude, db=True)
        net.name = name

        # Make sure kwargs are full like freq
        for k, v in kwargs.items():
            if isinstance(v, Quantity):
                kwargs[k] = xr.full_like(frequency, v)

        # Generate noise gain net if needed
        if noise_gain is not None:
            kwargs["noise_network"] = dgz.attenuator(noise_gain.to("dB").magnitude, db=True)

        super().__init__(name, manufacturer, pn, net, symbol=symbol, vsup=vsup, isup=isup, **kwargs)


class Cable(Device):
    """
    Device for defining a cable in terms of its length.
    Loss is given by adjusting tan_delta.
    """

    def __init__(
        self,
        name: str,
        manufacturer: str,
        pn: str,
        frequency: xr.DataArray,
        length: Quantity,
        tan_delta: float = 0.1,
        end_loss: Quantity = 0.015 * ureg.dB,
        z0: float = 50.0,
        **kwargs: Any,
    ) -> None:
        # Generate skrf Frequency object
        frequency = frequency.copy()
        frequency.data = frequency.data.to("Hz")
        freq = Frequency(frequency.values[0], frequency.values[-1], frequency.values.size, "hz")

        # Create media for end attenuation
        dgz = DefinedGammaZ0(freq, z0=z0)
        ends = dgz.attenuator(-end_loss.to("dB").magnitude, db=True)

        # Create coaxial cable
        coax = Coaxial(freq, z0=z0, tan_delta=tan_delta)

        # Cable is the cascade of ends with coax of specified length
        cable = ends ** coax.line(d=length.to("inch").magnitude, unit="in", name=name) ** ends
        cable.name = name

        if "symbol" not in kwargs:
            kwargs["symbol"] = Symbol(schemdraw.elements.cables.Coax, length=2)
        super().__init__(name, manufacturer, pn, cable, **kwargs)


class Mixer(MixerMixin, Device):
    def __init__(self, *args, lo_freq: float, sideband: str = "low", **kwargs):
        super().__init__(*args, lo_freq=lo_freq, sideband=sideband, **kwargs)


class MixerConstant(MixerMixin, Constant):
    def __init__(self, *args, lo_freq: float, sideband: str = "low", **kwargs):
        super().__init__(*args, lo_freq=lo_freq, sideband=sideband, **kwargs)
