"""Main module for xcvr."""

from __future__ import annotations

import base64
import warnings
from collections import Counter
from copy import copy, deepcopy

import markdown
import numpy as np
import schemdraw
import skrf.network
import xarray as xr
from loguru import logger
from pandas import DataFrame
from pandas.io.formats.style import Styler
from pint import Quantity, UnitStrippedWarning
from schemdraw import dsp
from skrf import Network
from xrench.units import ureg

from xcvr.conversions import nf2temp, temp2nf
from xcvr.frequency import FrequencyPlan, MixerMixin
from xcvr.symbols import Symbol
from xcvr.utils import net2da, xrnan2inf

# Suppress pint's warning when xr.where/assignment strips units from Quantity arrays.
warnings.filterwarnings("ignore", category=UnitStrippedWarning)
# Suppress scipy RuntimeWarning from interpolating through inf values (p1db, psat, oip3
# default to inf for passive devices). Values are correctly restored by xrnan2inf afterward.
warnings.filterwarnings("ignore", message="invalid value encountered", category=RuntimeWarning)

_xr_concat_kwargs = {"join": "outer", "coords": "all", "compat": "override"}


class Device:
    """Device object. Lowest level container of performance."""

    def __init__(
        self,
        name: str,
        manufacturer: str,
        pn: str,
        network: Network,
        noise_network: Network | None = None,
        symbol: Symbol | None = None,
        nf: xr.DataArray = None,
        oip3: xr.DataArray = None,
        p1db: xr.DataArray = None,
        psat: xr.DataArray = None,
        vsup: Quantity = 0 * ureg.volt,
        isup: Quantity = 0 * ureg.mA,
        t0: Quantity | None = None,
    ) -> None:
        # Store attributes
        self.name = name
        self.manufacturer = manufacturer
        self.pn = pn
        if t0 is None:
            t0 = ureg("290 kelvin")
        self.t0 = t0
        self.vsup = vsup.to("volt")
        self.isup = isup.to("mA")
        self.pdiss = (vsup * isup).to("mW")

        # Attributes to add to network DataArray
        # Store private variables
        self._network = network
        self._noise_network = noise_network
        self._nf = nf
        self._oip3 = oip3
        self._p1db = p1db
        self._psat = psat
        self.t0 = t0

        # Use empty box as symbol if not specified
        if symbol is None:
            symbol = Symbol(dsp.Square)
        self.symbol = symbol
        # Initialize frequency plan first, before interpolate touches anything
        self.freq_plan = FrequencyPlan.passthrough(self._network.f)
        # Interpolate to initialize
        self.interpolate(self.gain.frequency)

    @property
    def network(self) -> Network:
        """Signal scattering parameter data in a skrf.Network object format."""
        return self._network

    @property
    def noise_network(self) -> Network:
        """
        Network object data in a skrf.Network object format representing how noise is treated.
        This will differ for things like combiners.
        """
        if self._noise_network is None:
            self._noise_network = copy(self.network)
        return self._noise_network

    @property
    def s_mag_da(self) -> xr.DataArray:
        """Magnitude in decibels of the network as a DataArray."""
        return self._add_coords(net2da(self.network)).assign_coords(
            carrier_freq=("frequency", self.freq_plan.carrier),
        )

    @property
    def noise_s_mag_da(self) -> xr.DataArray:
        """Scattering parameter magnitude."""
        return self._add_coords(net2da(self.noise_network))

    @property
    def gain(self) -> xr.DataArray:
        """Gain (S21) of the device."""
        return self.s_mag_da.sel(out_port=1, in_port=0)

    @property
    def noise_gain(self) -> xr.DataArray:
        """Gain (S21) for noise of the device.
        This will differ for a combiner as noise will not add coherently.
        """
        return self.noise_s_mag_da.sel(out_port=1, in_port=0)

    @property
    def nf(self) -> xr.DataArray:
        """Device noise figure.
        If a passive device where nf has not been specified it will take the gain.
        """
        # If not specified get noise gain
        if self._nf is None:
            nf = 1 / self.noise_gain
            # Convert to dB
            nf.data = nf.data.to("dB")
            self._nf = nf
        self._nf = self._add_coords(self._nf)
        return self._nf

    @property
    def p1db(self) -> xr.DataArray:
        """Output referred 1 dB compression point."""
        if self._p1db is None:
            self._p1db = xr.full_like(self.gain, np.inf) * ureg.dBm
        self._p1db = self._add_coords(self._p1db)
        return self._p1db

    @property
    def psat(self) -> xr.DataArray:
        """Output referred saturated power level.
        If not specified, default is 1 dB higher than the P1dB compression point.
        """
        if self._psat is None:
            psat = self.p1db * (1 * ureg.dB)
            psat.data = psat.data.to("dBm")
            self._psat = psat
        self._psat = self._add_coords(self._psat)
        return self._psat

    @property
    def oip3(self) -> xr.DataArray:
        """Output referred third order intercept point."""
        if self._oip3 is None:
            self._oip3 = xr.full_like(self.gain, np.inf) * ureg.dBm
        self._oip3 = self._add_coords(self._oip3)
        return self._oip3

    @property
    def iip3(self) -> xr.DataArray:
        """Input referred third order intercept point."""
        iip3 = self.oip3 / self.gain
        iip3.data = iip3.data.to("dBm")
        return iip3

    @property
    def noise_temperature(self) -> Quantity:
        """Equivalent input noise temperature."""
        return nf2temp(self.nf, self.t0)

    def interpolate(self, frequency: xr.DataArray) -> None:
        """Interpolate self by interpolating the network objects."""
        # Convert frequency to hz to ensure units are correct
        frequency = frequency.copy()
        if isinstance(frequency.data, Quantity):
            frequency.data = frequency.data.to("Hz").magnitude

        # Determine kind of interpolation, if only a single frequency kind should be linear
        intrpkind = "cubic"
        linear_size = 4
        if frequency.size < linear_size or self._network.f.size < linear_size:
            intrpkind = "linear"
        # Interpolate the network objects
        self._network = self._network.interpolate(
            skrf.Frequency.from_f(frequency, unit="hz"),
            kind=intrpkind,
            fill_value="extrapolate",
        )
        if self._noise_network is not None:
            self._noise_network = self._noise_network.interpolate(
                skrf.Frequency.from_f(frequency, unit="hz"),
                kind=intrpkind,
                fill_value="extrapolate",
            )

        # Interpolate other parameters as necessary
        interp_attrs = ["nf", "oip3", "p1db", "psat"]
        for ia in interp_attrs:
            str_ia = f"_{ia}"
            attr = self.__getattribute__(str_ia)
            if isinstance(attr, Quantity):
                attr = xr.full_like(self.gain, attr.magnitude) * attr.units
            elif isinstance(attr, xr.DataArray):
                if isinstance(attr.data, Quantity):
                    units = attr.data.units
                    attr.data = attr.data.magnitude
                    attr = attr.interp(frequency=frequency, kwargs={"fill_value": "extrapolate"})
                    attr.data = attr.data * units
                else:
                    attr = attr.interp(frequency=frequency, kwargs={"fill_value": "extrapolate"})
                attr = xrnan2inf(attr)
            self.__setattr__(str_ia, attr)

    def _add_coords(self, da: xr.DataArray) -> xr.DataArray:
        """Add device, manufacturer, and pn as coords to DataArray."""
        dcs = {"device": self.name, "manufacturer": self.manufacturer, "pn": self.pn}
        return da.assign_coords(**dcs)

    def __repr__(self) -> str:
        """Device preview."""
        tab_props = ["Gain", "NF", "P1dB", "Psat", "OIP3", "IIP3"]
        encoded_sym = base64.b64encode(self.symbol()._repr_png_()).decode("utf-8")
        emb_sym = f"<img src='data:image/png;base64,{encoded_sym}'>"
        tab = "| Parameter | Value Fmin | Value Fmid | Value Fmax |\r\n"
        tab += "|---:|:---|:---|:---|\r\n"

        tab += f"| Frequency (MHz)| {self.gain.frequency[0].item() / 1e6}| {self.gain.frequency[int(self.gain.size / 2)].item() / 1e6}| {self.gain.frequency[-1].item() / 1e6} |\r\n"
        for tabp in tab_props:
            pval = self.__getattribute__(tabp.lower())
            tab += f"| {tabp} | {pval[0].item().magnitude:.2f} | {pval[int(pval.size / 2)].item().magnitude:.2f} | {pval[-1].item():.2f} |\r\n"
        tab += f"| Symbol | {emb_sym} |||\r\n"
        return tab

    def __str__(self) -> str:
        return self.__repr__()

    def _repr_html_(self) -> str:
        return markdown.markdown(self.__repr__(), extensions=["markdown.extensions.tables"])


class System:
    """Container of devices and systems to cascade."""

    def __init__(
        self,
        name: str,
        manufacturer: str,
        pn: str,
        devices: list[Device | System],
        symbol: Symbol | None = None,
        t0: Quantity | None = None,
        designator_append: str = "",
    ) -> None:
        # Store attributes
        self._devices = devices
        self._devices_cache: list[Device] | None = None
        self.name = name
        self.manufacturer = manufacturer
        self.pn = pn
        self.designator_append = designator_append

        # Initialize some variables
        self._expand = False
        # Use empty box as symbol if not specified
        if symbol is None:
            symbol = Symbol(dsp.Square)
        self.symbol = symbol

        # This get's propagated in _build_devices
        if t0 is None:
            t0 = ureg("290 kelvin")
        self.t0 = t0

    def get_device_attr(self, attr: str) -> xr.DataArray:
        """Pulls attribute from each device and returns a clean DataArray."""
        attrlist = []
        for d in self.devices:
            da = getattr(d, attr)
            da = d.freq_plan.label(da)
            attrlist.append(da)

        return xr.concat(attrlist, "device", **_xr_concat_kwargs)

    @property
    def devices(self) -> list[Device]:
        """List of devices."""
        if self._devices_cache is None:
            self._devices_cache = self._build_devices()
        return self._devices_cache

    def _build_devices(self) -> list[Device]:
        """
        Builds list of devices. Interpolates each device so all the frequencies
        are the same. Handles frequency translation at mixer boundaries via
        FrequencyPlan so RF/carrier axes stay explicit and unambiguous.
        """
        # Deep copy to avoid mutating the originals
        dvs = [deepcopy(d) for d in self._devices]

        # Flatten: expand sub-Systems or collapse them to a single Device
        dlist: list[Device] = []
        for dv in dvs:
            if isinstance(dv, System):
                dlist += dv.devices if dv.expand else [dv.as_device]
            elif isinstance(dv, Device):
                dlist.append(dv)
            else:
                raise TypeError(f"{dv} not a valid input.")

        # Split the flat list into bands, where each band ends at a mixer (or at the end)
        bands: list[list[Device]] = []
        current_band: list[Device] = []
        for dv in dlist:
            current_band.append(dv)
            if isinstance(dv, MixerMixin):
                bands.append(current_band)
                current_band = []
        if current_band:
            bands.append(current_band)

        # RF grid comes from the first device in the chain — this is the index,
        # fixed for the entire cascade
        rf_vals = bands[0][0].gain.frequency.values  # noqa: PD011
        rf_vals = rf_vals[~np.isnan(rf_vals)]
        freq_plan = FrequencyPlan.passthrough(rf_vals)

        flat: list[Device] = []
        for band in bands:
            has_mixer = isinstance(band[-1], MixerMixin)
            plain_devices = band[:-1] if has_mixer else band
            mixer = band[-1] if has_mixer else None

            for dv in plain_devices:
                dv.interpolate(freq_plan.carrier_da)
                dv.freq_plan = freq_plan
                dv.t0 = self.t0  # propagate system temperature down

                flat.append(dv)

            if mixer is not None:
                mixer.interpolate(freq_plan.rf_da)
                mixer.freq_plan = freq_plan.translate(
                    mixer.lo_freq.to("Hz").magnitude,
                    mixer.sideband,
                )
                flat.append(mixer)
                freq_plan = mixer.freq_plan

        return self._update_designators(flat)

    @property
    def expand(self) -> bool:
        """
        Whether to expand the system to show the individual
        components in a higher level system.
        """
        return self._expand

    @expand.setter
    def expand(self, expand: bool) -> None:
        if isinstance(expand, bool):
            self._expand = expand

    @property
    def as_device(self) -> Device:
        """Return self as a Device."""
        device = Device(
            self.name,
            self.manufacturer,
            self.pn,
            network=self.network,
            symbol=self.symbol,
            nf=self.nf,
            oip3=self.oip3,
            p1db=self.p1db,
            psat=self.psat,
            t0=self.t0,
        )
        device.pdiss = self.cascaded_pdiss.isel(device=-1).item()
        return device

    def _update_designators(self, devices: list[Device]) -> list[Device]:
        """Update designators to ensure they are unique."""
        # Check for duplicates in the name
        namecounts = dict(Counter([d.name for d in devices]))
        duplicates = {k: v for k, v in namecounts.items() if v > 1}
        for k in duplicates:
            i = 0
            for d in devices:
                if d.name == k:
                    # Update name
                    d.name = f"{d.name}{self.designator_append}{i}"
                    i += 1
        return devices

    @property
    def networks(self) -> list[Network]:
        """List of Network objects for each device in the system."""
        return [d.network for d in self.devices]

    @property
    def network(self) -> Network:
        """
        Cascaded S-parameters of the system. The frequency axis is the
        **output carrier frequency** (post-translation), not the RF input frequency.
        For a system without a mixer these are identical.
        """
        # Ensure the cascaded network is on the shared carrier grid
        rf_grid = self.devices[-1].freq_plan.carrier

        nets_scaled = []
        for net, d in zip(self.networks, self.devices, strict=False):
            if not d.freq_plan.is_translated:
                n = net.interpolate(
                    skrf.Frequency.from_f(d.freq_plan.rf, unit="hz"),
                    fill_value="extrapolate",
                )
                n.frequency = skrf.Frequency.from_f(rf_grid, unit="hz")
            else:
                n = net.interpolate(
                    skrf.Frequency.from_f(rf_grid, unit="hz"),
                    fill_value="extrapolate",
                )
            nets_scaled.append(n)
        net = skrf.network.cascade_list(nets_scaled)
        net.name = self.name
        return net

    @property
    def s_mag_da(self) -> xr.DataArray:
        """Scattering parameter magnitude, with carrier_freq coordinate attached."""
        da = self._add_coords(net2da(self.network))
        fp = self.devices[-1].freq_plan

        return da.assign_coords(carrier_freq=("frequency", fp.carrier))

    @property
    def gain(self) -> xr.DataArray:
        """Gain (S21) of the system."""
        gain = self.s_mag_da.sel(out_port=1, in_port=0)
        gain = gain.drop_vars(["in_port", "out_port"])
        gain.attrs = {
            **gain.attrs,
            "name": "Gain",
            "long_name": "Gain",
            "units": "dB",
            "description": "Gain",
        }
        return gain

    @property
    def vsup(self) -> xr.DataArray:
        """Device supply voltage."""
        return xr.DataArray(
            [d.vsup.to("volt").magnitude for d in self.devices] * ureg.volt,
            dims=("device",),
            coords={"device": [d.name for d in self.devices]},
        )

    @property
    def isup(self) -> xr.DataArray:
        """Device supply current."""
        return xr.DataArray(
            [d.isup.to("mA").magnitude for d in self.devices] * ureg.mA,
            dims=("device",),
            coords={"device": [d.name for d in self.devices]},
        )

    @property
    def pdiss(self) -> xr.DataArray:
        """Total power dissipation."""
        return xr.DataArray(
            [d.pdiss.to("mW").magnitude for d in self.devices] * ureg.mW,
            dims=("device",),
            coords={"device": [d.name for d in self.devices]},
        )

    @property
    def cascaded_pdiss(self) -> xr.DataArray:
        """Cascaded power dissipation (cumulative sum)."""
        return self.pdiss.cumsum(dim="device")

    @property
    def noise_networks(self) -> list[Network]:
        """
        List of Network objects for each device in the system representing how noise is treated.
        This will differ for things like combiners.
        """
        return [d.noise_network for d in self.devices]

    @property
    def noise_network(self) -> Network:
        """
        Cascaded of the networks in the system. Utilizes the s-parameters and
        skrf.network.cascade_list to perform the cascade.
        """
        return skrf.network.cascade_list(self.noise_networks)

    @property
    def cascaded_gain(self) -> xr.DataArray:
        """Returns the cumulative cascaded gain along the system."""
        cg = self.get_device_attr("gain").cumprod("device")
        cg.data = cg.data.to("dB")
        cg.attrs = {
            **cg.attrs,
            "name": "Gain",
            "long_name": "Gain",
            "units": "dB",
            "description": "Gain",
        }
        return cg

    @property
    def cascaded_noise_gain(self) -> xr.DataArray:
        """Returns the cumulative cascaded noise gain along the system."""
        cg = self.get_device_attr("noise_gain").cumprod("device")
        cg.data = cg.data.to("dB")
        cg.attrs = {
            **cg.attrs,
            "name": "Noise Gain",
            "long_name": "Noise Gain",
            "units": "dB",
            "description": "Noise Gain",
        }
        return cg

    @property
    def cascaded_noise_temperature(self) -> xr.DataArray:
        """Returns the cumulative cascaded equivalent input temperature
        along the system.
        """
        # Get noise gain data
        cng = self.cascaded_noise_gain
        # Roll
        cng = cng.shift({"device": 1})
        # Replace first item
        cng[{"device": 0}] = ureg("0 dB")

        # Cascade temperature and sum
        nt = self.get_device_attr("noise_temperature") / cng
        ntu = nt.data.units
        nt = nt.cumsum("device")
        nt.attrs = {
            **nt.attrs,
            "name": "Te",
            "long_name": "Noise Temperature",
            "units": ntu,
            "description": "Noise Temperature",
        }

        return nt

    @property
    def cascaded_nf(self) -> xr.DataArray:
        """Cascaded noise figure through the system."""
        with np.errstate(invalid="ignore", divide="ignore"):
            nf = temp2nf(self.cascaded_noise_temperature, self.t0)
            nf.data = nf.data.to("dB")
        nf.attrs = {
            **nf.attrs,
            "name": "NF",
            "long_name": "Noise Figure",
            "units": "dB",
            "description": "Noise Figure",
        }
        return nf

    def _cascade_intercept(self, attr: str, description: str) -> xr.DataArray:
        r"""
        Cascaded intercept point.

        The calculation follows the 'sum of reciprocals' method, assuming
        the worst-case scenario where intermodulation products add in-phase.

        .. math::

            \\frac{1}{OIP_{3,sys}} = \\sum_{i=1}^{n} \\left( \\frac{1}{OIP_{3,i} \\cdot \\prod_{j=i+1}^{n} G_j} \\right)

        Where:
        * :math:`OIP_{3,i}` is the linear OIP3 of the :math:`i`-th stage.
        * :math:`G_j` is the linear gain of the :math:`j`-th stage.
        """
        # Get gain and attr
        gain = self.get_device_attr("gain")
        inp = self.get_device_attr(attr)

        # Initialize
        cinp = [inp[{"device": 0}]]
        # Loop and compute cascade using the same formula as cascaded oip3
        for i in range(1, inp.device.size):
            # Calculate for current stage
            cinpn = 1 / (
                1 / (cinp[-1].data * gain[{"device": i}].data) + 1 / inp[{"device": i}].data
            )
            # Create DataArray
            cinpn = xr.DataArray(
                cinpn,
                dims=inp[{"device": i}].dims,
                coords=inp[{"device": i}].coords,
            )
            # Append
            cinp.append(cinpn)

        # Concatentate into a single DataArray
        cinp = xr.concat(
            cinp,
            "device",
            **_xr_concat_kwargs,
        )
        # Convert data units to dBm
        cinp.data = cinp.data.to("dBm")
        # Add in standard attrs
        cinp.attrs = {
            **cinp.attrs,
            "name": attr.upper(),
            "long_name": attr.upper(),
            "units": "dBm",
            "description": description,
        }

        return cinp

    @property
    def cascaded_p1db(self) -> xr.DataArray:
        """
        Cascaded 1dB compression point (P1dB) through the system.

        Method is same as cascading the third order intercept point.
        TODO: Add reference.
        """
        return self._cascade_intercept("p1db", "Output referred 1dB compression point")

    @property
    def cascaded_psat(self) -> xr.DataArray:
        """Cascaded saturated power point through the system."""
        # Get device psats
        dpsats = self.get_device_attr("psat")
        dgains = self.get_device_attr("gain")

        cpsat = [dpsats.isel(device=0)]

        for i in range(1, len(self.devices)):  # ← start at 1, not 0
            psat = dgains.isel(device=i) * cpsat[-1]
            dpsat = dpsats.isel(device=i).interp(
                frequency=psat.frequency,
                kwargs={"fill_value": "extrapolate"},
            )
            dpsat = xrnan2inf(dpsat)
            dpsat.data = dpsat.data * dpsats.isel(device=i).data.units
            psat = xr.where(psat > dpsat, dpsat, psat)
            cpsat.append(psat)

        cpsat = xr.concat(cpsat, "device", **_xr_concat_kwargs)
        cpsat = cpsat.assign_coords(
            device=("device", [d.name for d in self.devices]),
            carrier_freq=dpsats.carrier_freq,
        )
        cpsat.data = cpsat.data.to("dBm")
        cpsat.attrs = {
            **cpsat.attrs,
            "name": "Psat",
            "long_name": "Psat",
            "units": "dBm",
            "description": "Saturated Power",
        }
        return cpsat

    @property
    def cascaded_oip3(self) -> xr.DataArray:
        """Cascaded output referred third order intercept point (OIP3) through the system."""
        return self._cascade_intercept("oip3", "Output referred third order intercept point")

    @property
    def cascaded_iip3(self) -> xr.DataArray:
        """Cascaded input referred third order intercept point (IIP3) through the system."""
        dgain = self.get_device_attr("gain")
        dsz = dgain.device.size

        # Start from the output: IIP3 at last device = OIP3 / gain_last
        ciip3 = [
            self.oip3.drop_vars(["device", "manufacturer", "pn", "carrier_freq"], errors="ignore")
            / dgain.isel(device=-1).drop_vars("carrier_freq", errors="ignore"),
        ]

        # Walk backwards, dropping carrier_freq on each intermediate result
        for i in range(dsz)[-2::-1]:
            ciip3.append(
                ciip3[-1].drop_vars(
                    ["device", "manufacturer", "pn", "carrier_freq"],
                    errors="ignore",
                )
                / dgain.isel(device=i).drop_vars("carrier_freq", errors="ignore"),
            )

        # Reverse to device order, concat, then reattach carrier_freq cleanly
        ciip3 = xr.concat(
            ciip3[::-1],
            "device",
            **_xr_concat_kwargs,
        )
        ciip3 = ciip3.assign_coords(
            device=("device", [d.name for d in self.devices]),
            carrier_freq=dgain.carrier_freq,
        )

        ciip3.data = ciip3.data.to("dBm")
        ciip3.attrs = {
            **ciip3.attrs,
            "name": "IIP3",
            "long_name": "IIP3",
            "units": "dBm",
            "description": "Input referred third order intercept point",
        }
        return ciip3

    # Total System Properties
    @property
    def nf(self) -> xr.DataArray:
        """Total system noise figure (NF)."""
        return self._get_sys_prop("nf", "Noise Figure", "dB", "System Noise Figure")

    @property
    def noise_temperature(self) -> xr.DataArray:
        """Equivalent input noise temperature."""
        return nf2temp(self.nf, self.t0)

    @property
    def noise_gain(self) -> xr.DataArray:
        """Total system noise gain."""
        return self._get_sys_prop("noise_gain", "Noise Gain", "dB", "System Noise Gain")

    @property
    def p1db(self) -> xr.DataArray:
        """Total system output referred 1 dB compression point (P1dB)."""
        return self._get_sys_prop("p1db", "P1dB", "dBm", "Output referred 1dB compression point")

    @property
    def psat(self) -> xr.DataArray:
        """Total system output referred saturated power point (Psat)."""
        return self._get_sys_prop("psat", "Psat", "dBm", "Output referred saturated power point")

    @property
    def oip3(self) -> xr.DataArray:
        """Total system output referred third order intercept point (OIP3)."""
        oip3 = self._get_sys_prop(
            "oip3",
            "OIP3",
            "dBm",
            "Output referred third order intercept point",
        )
        return oip3

    @property
    def iip3(self) -> xr.DataArray:
        """
        Total system input referred third order intercept point (IIP3).
        TODO: Verify.
        """
        return self._get_sys_prop(
            "iip3",
            "IIP3",
            "dBm",
            "Input referred third order intercept point",
        )

    @property
    def input_p1db(self) -> xr.DataArray:
        """Total system input referred 1dB compression point."""
        ip1db = self.p1db / self.cascaded_gain[{"device": -1}]
        ip1db.data = ip1db.data.to("dBm")
        ip1db.attrs = {
            **ip1db.attrs,
            "name": "IP1dB",
            "long_name": "IP1dB",
            "units": "dBm",
            "description": "Input referred 1dB compression point",
        }
        return ip1db

    def block_diagram(
        self,
        input_dot: bool = True,
        output_dot: bool = True,
        **kwargs,
    ) -> schemdraw.Drawing:
        """
        Returns a block diagram representation of the System using schemdraw elements.
        **kwargs are passed to the schemdraw.Drawing object.
        """
        drwing = schemdraw.Drawing(**kwargs)
        elems = []

        if input_dot:
            elems.append(schemdraw.elements.Dot(radius=0.12, open=True))

        # A flat iterator of your fully processed devices (which have the updated designators)
        flat_device_iter = iter(self.devices)

        # Helper function to recursively traverse layers and record bounds
        def build_diagram_layers(current_system, depth=0) -> list[tuple[int, int, str, int]]:
            box_boundaries = []
            start_idx = len(elems)

            for d in current_system._devices:
                if isinstance(d, System) and d.expand:
                    # Traverses into the expanded subsystem
                    nested_boxes = build_diagram_layers(d, depth + 1)
                    box_boundaries.extend(nested_boxes)
                else:
                    # Pull the next processed device from our flat iterator to get the correct name
                    try:
                        processed_device = next(flat_device_iter)
                        display_name = processed_device.name
                        symbol_callable = processed_device.symbol
                    except StopIteration:
                        # Fallback case if the iterator runs dry safely
                        device_instance = d.as_device if isinstance(d, System) else d
                        display_name = device_instance.name
                        symbol_callable = device_instance.symbol

                    if not isinstance(symbol_callable(), dsp.Antenna):
                        elems.append(dsp.Line().length(drwing.unit / 4))

                    # Use display_name here (which includes the appended digit)
                    elem = symbol_callable().label(display_name, "top", ofst=0.2)
                    elems.append(elem)
                    elems.append(dsp.Line().length(drwing.unit / 4))

            end_idx = len(elems)

            # Record bounds only if it's an expanded system AND not the root level
            if current_system.expand and depth > 0:
                actual_start = start_idx + 1 if (start_idx == 1 and input_dot) else start_idx
                box_boundaries.append((actual_start, end_idx, current_system.name, depth))

            return box_boundaries

        # Gather boundaries containing (start, end, name, depth)
        all_boxes = build_diagram_layers(self, depth=0)

        if output_dot:
            elems.append(dsp.Line().length(drwing.unit / 4))
            elems.append(schemdraw.elements.Dot(radius=0.12, open=True))

        # Add flat schematic elements
        drwing.add_elements(*elems)

        # Draw the boxes
        if all_boxes:
            max_depth = max(box[3] for box in all_boxes)

            for start, end, name, depth in all_boxes:
                depth_offset = max_depth - depth

                # Snug horizontal layout
                padding_x = 0.08 + (depth_offset * 0.12)
                # Snug vertical layout keeping the bottom tight
                padding_y = 0.4 + (depth_offset * 0.5)
                # Float labels cleanly above lines
                label_offset = 0.1

                drwing += (
                    schemdraw.elements.lines.EncircleBox(
                        elems[start:end],
                        cornerradius=0.25,
                        padx=padding_x,
                        pady=padding_y,
                    )
                    .linestyle(":")
                    .label(name, loc="top", ofst=label_offset)
                )

        return drwing

    def get_dataset(self, input_power: Quantity | None = None, **kwargs) -> xr.Dataset:
        """
        Return the system as an xarray Dataset where the different performance
        parameters are attrs.
        """
        dattrskeys = [
            "gain",
            "noise_gain",
            "nf",
            "iip3",
            "oip3",
            "p1db",
            "psat",
            "noise_temperature",
        ]
        dattrs = {k: self.get_device_attr(k).sel(**kwargs) for k in dattrskeys}

        casc_attrs = {
            "cascaded_gain": self.cascaded_gain.sel(**kwargs),
            "cascaded_noise_gain": self.cascaded_noise_gain.sel(**kwargs),
            "cascaded_nf": self.cascaded_nf.sel(**kwargs),
            "cascaded_noise_temperature": self.cascaded_noise_temperature.sel(**kwargs),
            "cascaded_p1db": self.cascaded_p1db.sel(**kwargs),
            "cascaded_psat": self.cascaded_psat.sel(**kwargs),
            "cascaded_oip3": self.cascaded_oip3.sel(**kwargs),
            "cascaded_iip3": self.cascaded_iip3.sel(**kwargs),
            "vsup": self.vsup,
            "isup": self.isup,
            "pdiss": self.pdiss,
            "cascaded_pdiss": self.cascaded_pdiss,
        }

        if input_power is not None:
            casc_attrs["signal_level"] = self.get_signal_level(input_power).sel(**kwargs)

        ds = xr.Dataset({**dattrs, **casc_attrs})

        # # Reattach carrier_freq once, cleanly, on the assembled Dataset
        # carrier_freq = self.get_device_attr("gain").drop_vars("carrier_freq", errors="ignore")
        # carrier_freq = carrier_freq.sel(**kwargs) if kwargs else carrier_freq
        # ds = ds.assign_coords(carrier_freq=self.get_device_attr("gain").carrier_freq.sel(**kwargs))

        return ds

    def get_dataframe(self, input_power: Quantity | None = None, **kwargs) -> DataFrame:
        """Return the system as an pandas DataFrame."""
        # Get dataset
        ds = self.get_dataset(input_power=input_power, **kwargs)
        # Reset index to turn 'device' and 'frequency' into regular columns
        df = ds.to_dataframe().reset_index()
        # Replace the frequency index column with carrier_freq for readability
        if "carrier_freq" in df.columns:
            df["frequency"] = df["carrier_freq"]
            df = df.drop(columns=["carrier_freq"])
        # Format dataframe
        drop_cols = [
            "noise_gain",
            "noise_temperature",
            # "oip3",
            "iip3",
            "cascaded_noise_gain",
            "cascaded_noise_temperature",
            # "cascaded_oip3",
            # "cascaded_iip3",
            "in_port",
            "out_port",
        ]
        signal_present = input_power is not None

        return (
            df.drop(columns=drop_cols, errors="ignore")  # Added errors='ignore' for safety
            .round(2)
            .style.pipe(format_cascaded_table, self.name, signal_present=signal_present)
        )

    def get_signal_level(self, input_power: Quantity | xr.DataArray) -> xr.DataArray:
        """
        Get the cascaded signal level. Signal level will compress to Psat at each device.

        Parameters
        ----------
        input_power: Quantity, xr.DataArray
            Input power to the system

        Returns:
        -------
        output_power: xr.DataArray
            Cascaded output power at each device
        """
        # Output units will be in dBm
        units = ureg.dBm
        # Loop through each device, initialize with input power which will be removed later
        outpwrs = [input_power]
        gains = self.get_device_attr("gain")
        psats = self.get_device_attr("psat")
        for i in range(len(self.devices)):
            # Get output power from gain
            outpwr = gains.isel(device=i) * outpwrs[-1]
            # Set units
            outpwr.data = outpwr.data.to(units)

            # Get saturated power
            dpsat = psats.isel(device=i)
            dpsat = xrnan2inf(dpsat)

            # Limit to Psat
            outpwr = xr.where(outpwr > dpsat, dpsat, outpwr)
            # append
            outpwrs.append(outpwr)

        # Create a single DataArray, excluding the first record which is just the input signal
        outpwr = xr.concat(outpwrs[1:], "device", **_xr_concat_kwargs)
        # Convert the units
        outpwr.data = outpwr.data.to(units)
        # Set frequency
        outpwr = outpwr.assign_coords(carrier_freq=gains.carrier_freq)
        # Set attrs
        outpwr.attrs = {
            **outpwr.attrs,
            "name": "power",
            "long_name": "Power",
            "units": "dBm",
            "description": "Power",
        }
        return outpwr

    def _get_sys_prop(self, prop: str, long_name: str, unit: str, description: str) -> xr.DataArray:
        # Get cascaded attribute and select the value at the last device
        da = self._add_coords(self.__getattribute__(f"cascaded_{prop}")[{"device": -1}])
        # Drop input and output vars
        da = da.drop_vars(["in_port", "out_port"], errors="ignore")
        # Set units
        da.data = da.data.to(unit)
        # Set attributes that are helpful in plotting
        da.attrs = {
            **da.attrs,
            "name": prop,
            "long_name": long_name,
            "unit": unit,
            "description": description,
        }
        return da

    def _add_coords(self, da: xr.DataArray) -> xr.DataArray:
        """Add device, manufacturer, and pn as coords to DataArray."""
        dcs = {"device": self.name, "manufacturer": self.manufacturer, "pn": self.pn}

        return da.assign_coords(**dcs)

    def __repr__(self) -> str:
        """System preview."""
        tab_props = ["Gain", "NF", "P1dB", "Psat", "OIP3", "IIP3"]
        encoded_sym = base64.b64encode(self.symbol()._repr_png_()).decode("utf-8")
        emb_sym = f"<img src='data:image/png;base64,{encoded_sym}'>"
        tab = "| Parameter | Value Fmin | Value Fmid | Value Fmax |\r\n"
        tab += "|---:|:---|:---|:---|\r\n"

        tab += f"| Frequency (MHz)| {self.gain.frequency[0].item() / 1e6}| {self.gain.frequency[int(self.gain.size / 2)].item() / 1e6}| {self.gain.frequency[-1].item() / 1e6} |\r\n"
        for tabp in tab_props:
            pval = self.__getattribute__(tabp.lower())
            tab += f"| {tabp} | {pval[0].item().magnitude:.2f} | {pval[int(pval.size / 2)].item().magnitude:.2f} | {pval[-1].item():.2f} |\r\n"
        tab += f"| Symbol | {emb_sym} |||\r\n"
        return tab

    def __str__(self) -> str:
        return self.__repr__()

    def _repr_html_(self) -> str:
        return markdown.markdown(self.__repr__(), extensions=["markdown.extensions.tables"])


def highlight_compression(s: DataFrame) -> list[str]:
    """Adds highlight to tabular DataFrame view for signal compression."""
    highlight = "background-color: #ffffb3;"
    # Use explicit names instead of s[0], s[1], s[2]
    sig = s["signal_level"]
    p1 = s["cascaded_p1db"]
    psat = s["cascaded_psat"]

    styles = [""] * 3  # for [signal_level, cascaded_p1db, cascaded_psat]

    if sig >= psat:
        styles[0] = "color:red;" + highlight
        styles[2] = highlight
    elif sig >= p1:
        styles[0] = "color:red;" + highlight
        styles[1] = highlight

    return styles


def format_cascaded_table(styler: Styler, caption: str, signal_present: bool = True) -> Styler:
    """Custom formatting of pandas DataFrame using the Styler."""
    # Set caption
    styler.set_caption(caption)

    # Column number formats
    fmt_dict = {
        "frequency": "{:g}",
        "gain": "{:.2f}",
        "nf": "{:.2f}",
        "p1db": "{:.2f}",
        "psat": "{:.2f}",
        "cascaded_gain": "{:.2f}",
        "cascaded_nf": "{:.2f}",
        "cascaded_p1db": "{:.2f}",
        "cascaded_psat": "{:.2f}",
    }

    # Add a background bar to the cascaded NF and gain columns to show contribution to total
    styler.bar(subset=["cascaded_nf"], color="#78BBE8")
    styler.bar(subset=["cascaded_gain"], align="mid", color=["#FFBB7D", "#78BBE8"])

    # Do the same for signal level if input power is given
    if signal_present:
        # Add bar
        styler.bar(subset=["signal_level"], align="mid", color="#78BBE8")
        # Add number format to the column
        fmt_dict = {**fmt_dict, "signal_level": "{:.2f}"}
        slice_ = ["signal_level", "cascaded_p1db", "cascaded_psat"]
        styler.apply(highlight_compression, axis=1, subset=slice_)

    # Apply format
    styler.format(fmt_dict)
    return styler
