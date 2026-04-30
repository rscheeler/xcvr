"""Main module"""

import base64
import warnings
from collections import Counter
from copy import copy, deepcopy
from typing import Optional, Union

import markdown
import numpy as np
import schemdraw
import skrf.network
import xarray as xr
from pandas import DataFrame
from pandas.io.formats.style import Styler
from pint import Quantity, UnitStrippedWarning
from schemdraw import dsp
from skrf import Network
from xrench.units import ureg

from .conversions import nf2temp, temp2nf
from .symbols import Symbol
from .utils import net2da, xrnan2inf

# Suppress pint's warning when xr.where/assignment strips units from Quantity arrays.
warnings.filterwarnings("ignore", category=UnitStrippedWarning)
# Suppress scipy RuntimeWarning from interpolating through inf values (p1db, psat, oip3
# default to inf for passive devices). Values are correctly restored by xrnan2inf afterward.
warnings.filterwarnings("ignore", message="invalid value encountered", category=RuntimeWarning)


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
        nf=None,
        oip3=None,
        p1db=None,
        psat=None,
        vsup: Quantity = 0 * ureg.volt,
        isup: Quantity = 0 * ureg.mA,
        t0: Quantity = ureg("290 kelvin"),
    ) -> None:
        # Store attributes
        self.name = name
        self.manufacturer = manufacturer
        self.pn = pn
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
        """Scattering parameter magnitude."""
        da = self._add_coords(net2da(self.network))
        return da

    @property
    def noise_s_mag_da(self) -> xr.DataArray:
        """Scattering parameter magnitude."""
        da = self._add_coords(net2da(self.noise_network))
        return da

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

    def interpolate(self, frequency) -> None:
        """Interpolate self by interpolating the network objects."""
        # Convert frequency to hz to ensure units are correct
        frequency = frequency.copy()
        if isinstance(frequency.data, Quantity):
            frequency.data = frequency.data.to("Hz").magnitude

        # Determine kind of interpolation, if only a single frequency kind should be linear
        intrpkind = "cubic"
        if frequency.size == 1 or self._network.f.size == 1:
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
            ia = f"_{ia}"
            attr = self.__getattribute__(ia)
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
            self.__setattr__(ia, attr)

    def _add_coords(self, da: xr.DataArray) -> xr.DataArray:
        """Add device, manufacturer, and pn as coords to DataArray."""
        dcs = {"device": self.name, "manufacturer": self.manufacturer, "pn": self.pn}
        da = da.assign_coords(**dcs)
        return da

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
        devices: list[Union[Device, "System"]],
        symbol: Symbol | None = None,
        t0: Quantity = ureg("290 kelvin"),
        designator_append: str = "",
    ) -> None:
        # Store attributes
        self._devices = devices
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

        # TODO: pass down to devices
        self.t0 = t0

    def get_device_attr(self, attr) -> xr.DataArray:
        """Pulls attribute from each device and returns a DataArray."""
        attrlist = [d.__getattribute__(attr) for d in self.devices]
        attr = xr.concat(
            attrlist,
            "device",
            join="outer",
            coords="minimal",
            compat="override",
        )
        return attr

    @property
    def devices(self) -> list[Device]:
        """
        List of devices. Interpolates each device so all the frequencies are the same, also need
        to deal with Systems vs. Devices differently.
        """
        # Copy items in devices list
        dvs = [deepcopy(d) for d in self._devices]

        # Create a device for the subsystem or grab the individual
        dlist = []
        for dv in dvs:
            # Handle a System object different than devices
            if isinstance(dv, System):
                if dv.expand:
                    dlist += dv.devices
                else:
                    dlist.append(dv.as_device)
            elif isinstance(dv, Device):
                dlist.append(dv)
            else:
                raise ValueError(f"{dv} not a valid input.")

        # Get the union of frequencies by getting the gain frequency dimension
        fs = xr.concat(
            [d.gain.frequency for d in dlist],
            "device",
            join="outer",
            coords="minimal",
            compat="override",
        )
        fs = np.unique(fs)
        # Drop nans
        fs = fs[np.logical_not(np.isnan(fs))]
        # Interpolate all the devices with the new frequency
        for dv in dlist:
            dv.interpolate(fs)
        # Get unique designators
        devices = self._update_designators(dlist)

        return devices

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
        for k, v in duplicates.items():
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
        nets = [d.network for d in self.devices]
        return nets

    @property
    def network(self) -> Network:
        """
        Cascaded of the networks in the system. Utilizes the s-parameters and skrf.network.cascade_list
        to perform the cascade.
        """
        net = skrf.network.cascade_list(self.networks)
        return net

    @property
    def s_mag_da(self) -> xr.DataArray:
        """Scattering parameter magnitude."""
        da = self._add_coords(net2da(self.network))
        return da

    @property
    def gain(self) -> xr.DataArray:
        """Gain (S21) of the system."""
        gain = self.s_mag_da.sel(out_port=1, in_port=0)
        gain = gain.drop_vars(["in_port", "out_port"])
        gain.attrs = {
            **gain.attrs,
            **dict(name="Gain", long_name="Gain", units="dB", description="Gain"),
        }
        return gain

    @property
    def vsup(self) -> xr.DataArray:
        """Device supply voltage."""
        vsup = xr.DataArray(
            [d.vsup.to("volt").magnitude for d in self.devices] * ureg.volt,
            dims=("device",),
            coords=dict(device=[d.name for d in self.devices]),
        )
        return vsup

    @property
    def isup(self) -> xr.DataArray:
        """Device supply current."""
        isup = xr.DataArray(
            [d.isup.to("mA").magnitude for d in self.devices] * ureg.mA,
            dims=("device",),
            coords=dict(device=[d.name for d in self.devices]),
        )
        return isup

    @property
    def pdiss(self) -> xr.DataArray:
        """Total power dissipation."""
        pdiss = xr.DataArray(
            [d.pdiss.to("mW").magnitude for d in self.devices] * ureg.mW,
            dims=("device",),
            coords={"device": [d.name for d in self.devices]},
        )
        return pdiss

    @property
    def cascaded_pdiss(self) -> xr.DataArray:
        cpdiss = self.pdiss.cumsum(dim="device")
        return cpdiss

    @property
    def noise_networks(self) -> list[Network]:
        """
        List of Network objects for each device in the system representing how noise is treated.
        This will differ for things like combiners.
        """
        nets = [d.noise_network for d in self.devices]
        return nets

    @property
    def noise_network(self) -> Network:
        """
        Cascaded of the networks in the system. Utilizes the s-parameters and skrf.network.cascade_list
        to perform the cascade.
        """
        net = skrf.network.cascade_list(self.noise_networks)
        return net

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
        cng = cng.shift(dict(device=1))
        # Replace first item
        cng[dict(device=0)] = ureg("0 dB")

        # Cascade temperature and sum
        nt = self.get_device_attr("noise_temperature") / cng
        ntu = nt.data.units
        nt = nt.cumsum("device")
        nt.attrs = {
            **nt.attrs,
            **dict(
                name="Te",
                long_name="Noise Temperature",
                units=ntu,
                description="Noise Temperature",
            ),
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
            **dict(
                name="NF",
                long_name="Noise Figure",
                units="dB",
                description="Noise Figure",
            ),
        }
        return nf

    @property
    def cascaded_p1db(self) -> xr.DataArray:
        """
        Cascaded 1dB compression point (P1dB) through the system.

        Method is same as cascading the third order intercept point.
        TODO: Add reference.
        """
        # Get gain and p1dbs
        gain = self.get_device_attr("gain")
        p1db = self.get_device_attr("p1db")

        # Initialize
        cp1db = [p1db[dict(device=0)]]
        # Loop and compute cascade using the same formula as cascaded oip3
        for i in range(1, p1db.device.size):
            # Calculate for current stage
            cp1dbn = 1 / (
                1 / (cp1db[-1].data * gain[dict(device=i)].data) + 1 / p1db[dict(device=i)].data
            )
            # Create DataArray
            cp1dbn = xr.DataArray(
                cp1dbn,
                dims=p1db[dict(device=i)].dims,
                coords=p1db[dict(device=i)].coords,
            )
            # Append
            cp1db.append(cp1dbn)

        # Concatentate into a single DataArray
        cp1db = xr.concat(
            cp1db,
            "device",
            join="outer",
            coords="minimal",
            compat="override",
        )
        # Convert data units to dBm
        cp1db.data = cp1db.data.to("dBm")
        # Add in standard attrs
        cp1db.attrs = {
            **cp1db.attrs,
            **dict(
                name="P1dB",
                long_name="P1dB",
                units="dBm",
                description="Output referred 1dB compression point",
            ),
        }
        return cp1db

    @property
    def cascaded_psat(self) -> xr.DataArray:
        """Cascaded saturated power point through the system."""
        # Initialize
        cpsat = [self.devices[0].psat]
        # Loop over devices
        for d in self.devices[1:]:
            # Get psat[-1]*gain
            psat = d.gain * cpsat[-1]
            # Interpolate
            dpsat = d.psat.interp(
                frequency=psat.frequency,
                kwargs={"fill_value": "extrapolate"},
            )
            dpsat = xrnan2inf(dpsat)
            # Set Units
            dpsat.data = dpsat.data * d.psat.data.units
            # Limit to Psat
            psat = xr.where(psat > dpsat, dpsat, psat)
            # Append
            cpsat.append(psat)

        # Create a DataArray
        cpsat = xr.concat(
            cpsat,
            "device",
            join="outer",
            coords="minimal",
            compat="override",
        )
        # Convert units
        cpsat.data = cpsat.data.to("dBm")
        # Set attrs
        cpsat.attrs = {
            **cpsat.attrs,
            **dict(
                name="Psat",
                long_name="Psat",
                units="dBm",
                description="Saturated Power",
            ),
        }
        return cpsat

    @property
    def cascaded_oip3(self) -> xr.DataArray:
        """Cascaded output referred third order intercept point (OIP3) through the system."""
        # Get gain and oip3
        gain = self.get_device_attr("gain")
        oip3 = self.get_device_attr("oip3")

        # Initialize
        coip3 = [oip3[dict(device=0)]]
        # Loop and compute cascade using the same formula as cascaded oip3
        for i in range(1, oip3.device.size):
            # Calculate for current stage
            coip3n = 1 / (
                1 / (coip3[-1].data * gain[dict(device=i)].data) + 1 / oip3[dict(device=i)].data
            )
            # Create DataArray
            coip3n = xr.DataArray(
                coip3n,
                dims=oip3[dict(device=i)].dims,
                coords=oip3[dict(device=i)].coords,
            )
            # Append
            coip3.append(coip3n)

        # Concatentate into a single DataArray
        coip3 = xr.concat(
            coip3,
            "device",
            join="outer",
            coords="minimal",
            compat="override",
        )
        # Convert data units to dBm
        coip3.data = coip3.data.to("dBm")
        # Add in standard attrs
        coip3.attrs = {
            **coip3.attrs,
            **dict(
                name="OIP3",
                long_name="OIP3",
                units="dBm",
                description="Output referred third order intercept point",
            ),
        }
        return coip3

    @property
    def cascaded_iip3(self) -> xr.DataArray:
        """
        Cascaded input referred third order intercept point (IIP3) through the system
        TODO: Verify.
        """
        dgain = self.get_device_attr("gain")
        ciip3 = [self.oip3.drop_vars(["device", "manufacturer", "pn"]) / dgain.isel(device=-1)]
        dsz = dgain.device.size
        for i in range(dsz)[-2::-1]:
            ciip3.append(
                ciip3[-1].drop_vars(["device", "manufacturer", "pn"]) / dgain.isel(device=i),
            )
        ciip3 = xr.concat(
            ciip3[::-1],
            "device",
            join="outer",
            coords="minimal",
            compat="override",
        )
        ciip3.data = ciip3.data.to("dBm")

        return ciip3

    # Total System Properties
    @property
    def nf(self) -> xr.DataArray:
        """Total system noise figure (NF)."""
        nf = self._get_sys_prop("nf", "Noise Figure", "dB", "System Noise Figure")
        return nf

    @property
    def noise_temperature(self) -> xr.DataArray:
        """Equivalent input noise temperature."""
        return nf2temp(self.nf, self.t0)

    @property
    def noise_gain(self) -> xr.DataArray:
        """Total system noise gain."""
        ng = self._get_sys_prop("noise_gain", "Noise Gain", "dB", "System Noise Gain")
        return ng

    @property
    def p1db(self) -> xr.DataArray:
        """Total system output referred 1 dB compression point (P1dB)."""
        p1db = self._get_sys_prop("p1db", "P1dB", "dBm", "Output referred 1dB compression point")
        return p1db

    @property
    def psat(self) -> xr.DataArray:
        """Total system output referred saturated power point (Psat)."""
        psat = self._get_sys_prop("psat", "Psat", "dBm", "Output referred saturated power point")
        return psat

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
        iip3 = self.oip3 / self.gain
        iip3.data = iip3.data.to("dBm")
        iip3 = self._get_sys_prop(
            "iip3",
            "IIP3",
            "dBm",
            "Input referred third order intercept point",
        )
        return iip3

    @property
    def input_p1db(self) -> xr.DataArray:
        """Total system input referred 1dB compression point."""
        ip1db = self.p1db / self.cascaded_gain[dict(device=-1)]
        ip1db.data = ip1db.data.to("dBm")
        ip1db.attrs = {
            **ip1db.attrs,
            **dict(
                name="IP1dB",
                long_name="IP1dB",
                units="dBm",
                description="Input referred 1dB compression point",
            ),
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
        # Connect elements
        drwing = schemdraw.Drawing(**kwargs)
        # Initialize with a dot and open line
        elems = []
        i = 1
        if input_dot:
            elems.append(schemdraw.elements.Dot(radius=0.12, open=True))
            i += 1
        # Loop over devices
        for d in self.devices:
            # Create element instance
            elem = d.symbol().label(d.name, "top", ofst=0.2)
            # Short line on input, but format depends on if it is an antenna
            if not isinstance(elem, dsp.Antenna):
                inputline = dsp.Line().length(drwing.unit / 4)
                elems.append(inputline)
            else:
                i -= 1
            # Add device symbol
            elems.append(elem)
            # Short line on output
            elems.append(dsp.Line().length(drwing.unit / 4))

        # Output dot
        if output_dot:
            elems.append(dsp.Line().length(drwing.unit / 4))
            elems.append(schemdraw.elements.Dot(radius=0.12, open=True))

        # Add all the elements to the drawing
        drwing.add_elements(*elems)

        # Add subsystem boxes
        for d in self._devices:
            # Handle systems different than devices
            if isinstance(d, System):
                if d.expand:
                    drwing += (
                        schemdraw.elements.lines.EncircleBox(
                            elems[slice(i, i + len(d.devices) * 3 - 1)],
                            cornerradius=0.3,
                            padx=0,
                            pady=1,
                        )
                        .linestyle(":")
                        .label(d.name)
                    )
                    i += (len(d.devices) - 1) * 3
            i += 3

        return drwing

    def get_dataset(self, input_power: Quantity | None = None, **kwargs) -> xr.Dataset:
        """
        Return the system as an xarray Dataset where the different performance parameters are attrs.
        """
        # Get device attrs
        dattrs = dict()
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
        for k in dattrskeys:
            dattrs[k] = self.get_device_attr(k).sel(**kwargs)

        # Get cascaded attrs
        casc_attrs = dict(
            cascaded_gain=self.cascaded_gain.sel(**kwargs),
            cascaded_noise_gain=self.cascaded_noise_gain.sel(**kwargs),
            cascaded_nf=self.cascaded_nf.sel(**kwargs),
            cascaded_noise_temperature=self.cascaded_noise_temperature.sel(**kwargs),
            cascaded_p1db=self.cascaded_p1db.sel(**kwargs),
            cascaded_psat=self.cascaded_psat.sel(**kwargs),
            cascaded_oip3=self.cascaded_oip3.sel(**kwargs),
            cascaded_iip3=self.cascaded_iip3.sel(**kwargs),
            vsup=self.vsup,
            isup=self.isup,
            pdiss=self.pdiss,
            cascaded_pdiss=self.cascaded_pdiss,
        )
        # Add signal level if input power specified
        if input_power is not None:
            siglev = self.get_signal_level(input_power).sel(**kwargs)
            casc_attrs["signal_level"] = siglev

        return xr.Dataset({**dattrs, **casc_attrs})

    def get_dataframe(self, input_power: Quantity | None = None, **kwargs) -> DataFrame:
        """Return the system as an pandas DataFrame."""
        # Get dataset
        ds = self.get_dataset(input_power=input_power, **kwargs)
        # Reset index to turn 'device' and 'frequency' into regular columns
        df = ds.to_dataframe().reset_index()
        # Format dataframe
        drop_cols = [
            "noise_gain",
            "noise_temperature",
            "oip3",
            "iip3",
            "cascaded_noise_gain",
            "cascaded_noise_temperature",
            # "cascaded_oip3",
            # "cascaded_iip3",
            "in_port",
            "out_port",
        ]
        signal_present = input_power is not None
        df = (
            df.drop(columns=drop_cols, errors="ignore")  # Added errors='ignore' for safety
            .round(2)
            .style.pipe(format_cascaded_table, self.name, signal_present=signal_present)
        )
        return df

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
        for d in self.devices:
            # Get output power from gain
            outpwr = d.gain * outpwrs[-1]
            # Set units
            outpwr.data = outpwr.data.to(units)
            # Get saturated power
            dpsat = d.psat.interp(
                frequency=outpwr.frequency,
                kwargs={"fill_value": "extrapolate"},
            )
            dpsat = xrnan2inf(dpsat)
            dpsat.data = dpsat.data * d.psat.data.units

            # Limit to Psat
            outpwr = xr.where(outpwr > dpsat, dpsat, outpwr)
            # append
            outpwrs.append(outpwr)

        # Create a single DataArray, excluding the first record which is just the input signal
        outpwr = xr.concat(
            outpwrs[1:],
            "device",
            join="outer",
            coords="minimal",
            compat="override",
        )
        # Convert the units
        outpwr.data = outpwr.data.to(units)
        # Set attrs
        outpwr.attrs = {
            **outpwr.attrs,
            **dict(name="power", long_name="Power", units="dBm", description="Power"),
        }
        return outpwr

    def _get_sys_prop(self, prop: str, long_name: str, unit: str, description: str) -> xr.DataArray:
        # Get cascaded attribute and select the value at the last device
        da = self._add_coords(self.__getattribute__(f"cascaded_{prop}")[dict(device=-1)])
        # Drop input and output vars
        da = da.drop_vars(["in_port", "out_port"])
        # Set units
        da.data = da.data.to(unit)
        # Set attributes that are helpful in plotting
        da.attrs = {
            **da.attrs,
            **dict(name=prop, long_name=long_name, unit=unit, description=description),
        }
        return da

    def _add_coords(self, da: xr.DataArray) -> xr.DataArray:
        """
        Add device, manufacturer, and pn as coords to DataArray.
        """
        dcs = dict(device=self.name, manufacturer=self.manufacturer, pn=self.pn)
        da = da.assign_coords(**dcs)
        return da

    def __repr__(self) -> str:
        """
        System preview
        """
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
    """
    Custom formatting of pandas DataFrame using the Styler
    """
    # Set caption
    styler.set_caption(caption)

    # Column number formats
    fmt_dict = dict(
        frequency="{:g}",
        gain="{:.2f}",
        nf="{:.2f}",
        p1db="{:.2f}",
        psat="{:.2f}",
        cascaded_gain="{:.2f}",
        cascaded_nf="{:.2f}",
        cascaded_p1db="{:.2f}",
        cascaded_psat="{:.2f}",
    )

    # Add a background bar to the cascaded NF and gain columns to show contribution to total
    styler.bar(subset=["cascaded_nf"], color="#78BBE8")
    styler.bar(subset=["cascaded_gain"], align="mid", color=["#FFBB7D", "#78BBE8"])

    # Do the same for signal level if input power is given
    if signal_present:
        # Add bar
        styler.bar(subset=["signal_level"], align="mid", color="#78BBE8")
        # Add number format to the column
        fmt_dict = {**fmt_dict, **dict(signal_level="{:.2f}")}
        slice_ = ["signal_level", "cascaded_p1db", "cascaded_psat"]
        styler.apply(highlight_compression, axis=1, subset=slice_)

    # Apply format
    styler.format(fmt_dict)
    return styler
