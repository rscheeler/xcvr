from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from itertools import islice
from typing import Union

import numpy as np
from pint import Quantity
from xarray import DataArray
from xrench.units import ureg

from xcvr import ber


def nf2temp(nf: Quantity | DataArray, t0: Quantity) -> Quantity | DataArray:
    """
    Converts noise figure to noise temperature.

    Parameters
    ----------
    nf : Quantity, DataArray
        Noise figure
    t0 : Quantity, DataArray
        Reference temperature

    Returns:
    -------
    te : Quantity, DataArray
        Equivalent input noise temperature.
    """
    return t0 * (nf - 1)


@dataclass()
class LinkBudget:
    """Calculate link budget based off input parameters. Will attempt to solve for parameters not given."""

    name: str
    pt: Quantity
    gt: Quantity
    gr: Quantity
    nf: Quantity
    distance: Quantity
    frequency: Quantity
    bandwidth: Quantity
    data_rate: Quantity
    ber_req: float
    modulation: str
    modulation_order: int
    t0: Quantity = 290 * ureg.kelvin
    ta: Quantity = 290 * ureg.kelvin
    additional_loss: Quantity = 0 * ureg.dB
    coding_gain: Quantity = 0 * ureg.dB

    def __repr__(self) -> str:
        table = f"| Parameter | {self.name} |\r\n"
        table += "|:--:|:--:|\r\n"
        rows = asdict(self)
        rows.pop("name")
        for k, v in rows.items():
            vfmt = v
            if isinstance(v, (float, Quantity)):
                vfmt = f"{v:.3g}"
            table += f"| {k} | {vfmt}|\r\n"
        return table

    def _repr_markdown_(self) -> str:
        return self.__repr__()

    def view_link_table(self, link: str = "tabular_margin", markdown: bool = False) -> str:
        """View formatted link budget table for specified link."""
        if markdown:
            table = f"| Parameter |{self.name} {link} | Unit |\r\n"
            table += "|-:|:-:|:-|\r\n"
            link_dictionary = getattr(self, link)

            for k, v in islice(link_dictionary.items(), 0, len(link_dictionary) - 1):
                if not isinstance(v, Quantity):
                    v = Quantity(v * 1.0)
                table += f"|{k}|{v.magnitude:0.2f}|{v.units}|\r\n"

            table += f"|{list(link_dictionary.keys())[-1]}|{list(link_dictionary.values())[-1].magnitude:0.2f}|{list(link_dictionary.values())[-1].units}|\r\n"

        else:
            table = f"<table><caption>{self.name} {link} Link Budget</caption><tr><thead><th>Parameter</th><th>Value</th><th style='text-align:left'>Unit</th></thead></tr>\r\n<tbody>"

            link_dictionary = getattr(self, link)

            for k, v in islice(link_dictionary.items(), 0, len(link_dictionary) - 1):
                if not isinstance(v, Quantity):
                    v = Quantity(v * 1.0)
                table += f"<tr><th>{k}</th><td>{v.magnitude:0.2f}</td><td style='text-align:left'>{v.units}</td></tr>\r\n"

            table += f"<tr style='border-top: solid 1px'><th>{list(link_dictionary.keys())[-1]}</th><td>{list(link_dictionary.values())[-1].magnitude:0.2f}</td><td style='text-align:left'>{list(link_dictionary.values())[-1].units}</td></tr>\r\n"

            table += "</tbody></table>"
        return table

    @property
    def wavelength(self) -> Quantity:
        """Return the wavelength in meters."""
        return (1 / self.frequency * ureg.speed_of_light).to_base_units()

    @property
    def path_loss(self) -> Quantity:
        """
        Return free space path loss based off of range. If range given calculate from range and
        wavelength. If not given, solve for.
        """
        if self.distance is None:
            raise NotImplementedError
        pl = (self.distance * 4 * np.pi / self.wavelength).to_base_units() ** 2
        return pl.to("dB")

    @property
    def eirp(self) -> Quantity:
        """
        Returns the effective isotropic radiated power (EIRP) which is the transmit power times the
        transmit antenna gain.
        """
        return self.pt * self.gt

    @property
    def tsys(self) -> Quantity:
        r"""
        Returns the system temperature. Note it is assumed that the antenna radiometric temperature
        is equal to the reference temperature t0, resulting in the following.
        .. math::
            T_{sys}=T_{A'}+T_{rec},
            T_{\mathrm{sys}} = T_0+T_0(F-1)=T_0 F.


        """
        return self.ta + nf2temp(self.nf, self.t0)

    @property
    def rxnp(self) -> Quantity:
        """Return the receiver sensitivity or noise power."""
        return (ureg.boltzmann_constant * self.tsys * self.bandwidth).to("dBm")

    @property
    def rxpwr(self) -> Quantity:
        """Returns the received power."""
        return (self.pt * self.gt * self.gr / (self.path_loss * self.additional_loss)).to("dBm")

    @property
    def minrxp(self) -> Quantity:
        """Minimum received power."""
        return (self.pt * self.gt * self.gr / (self.max_path_loss * self.additional_loss)).to("dBm")

    @property
    def g_over_t(self) -> Quantity:
        """Return the receiver gain over system temperature ratio."""
        return self.gr / self.tsys

    @property
    def cnr(self) -> Quantity:
        """Returns the carrier-to-noise ratio."""
        return (
            self.eirp
            * self.g_over_t
            * self.coding_gain
            / (self.path_loss * ureg.boltzmann_constant * self.bandwidth * self.additional_loss)
        ).to("dB")

    @property
    def ebno(self) -> Quantity:
        """Returns the energy per bit to noise power spectral density ratio."""
        return (
            self.eirp
            * self.g_over_t
            * self.coding_gain
            / (self.path_loss * ureg.boltzmann_constant * self.data_rate * self.additional_loss)
        ).to("dB")

    @property
    def ebno_req(self) -> Quantity:
        """Returns the required energy per bit to noise power spectral density ratio to meet the
        link based off inputs.
        """
        return getattr(ber, f"m{self.modulation}_ebno")(self.modulation_order, self.ber_req)

    @property
    def link_margin(self) -> Quantity:
        """Returns margin for specified link."""
        return (self.ebno / self.ebno_req).to("dB")

    @property
    def ber(self) -> Quantity:
        """Returns the bit-error-rate from the ebno."""
        return getattr(ber, f"m{self.modulation}_ber")(self.modulation_order, self.ebno)

    @property
    def max_path_loss(self) -> Quantity:
        """Returns the maximum path loss to still meet link."""
        return (
            self.eirp
            * self.g_over_t
            * self.coding_gain
            / (self.ebno_req * ureg.boltzmann_constant * self.data_rate * self.additional_loss)
        ).to("dB")

    @property
    def max_link_distance(self) -> Quantity:
        """Returns the maximum link distance."""
        return (np.sqrt(self.max_path_loss.to_base_units()) * self.wavelength / (4 * np.pi)).to(
            "km",
        )

    @property
    def tabular_cnr(self) -> OrderedDict:
        """Returns an ordered dictionary of parameters for tabular display of the CNR link
        budget.
        """
        numer = {
            "pt": self.pt.to("dBm"),
            "gt": self.gt,
            "gr": self.gr,
            "coding_gain": self.coding_gain,
        }
        denom = {
            "path_loss": self.path_loss,
            "additional_loss": self.additional_loss,
            "k": (1 * ureg.boltzmann_constant).to(ureg.dBm / ureg.kelvin / ureg.Hz),
            "tsys": self.tsys.to("decibelkelvin"),
            "bandwidth": self.bandwidth.to("dBHz"),
        }
        return OrderedDict({**numer, **denom, "cnr": dictionary_ratio(numer, denom).to("dB")})

    @property
    def tabular_ebno(self) -> OrderedDict:
        """Returns an ordered dictionary of parameters for tabular display of the EBNO link budget."""
        numer = {
            "pt": self.pt.to("dBm"),
            "gt": self.gt,
            "gr": self.gr,
            "coding_gain": self.coding_gain,
        }
        denom = {
            "path_loss": self.path_loss,
            "additional_loss": self.additional_loss,
            "k": (1 * ureg.boltzmann_constant).to(ureg.dBm / ureg.kelvin / ureg.Hz),
            "tsys": self.tsys.to("decibelkelvin"),
            "data_rate": self.data_rate.to("decibelhertz"),
        }
        return OrderedDict({**numer, **denom, "ebno": dictionary_ratio(numer, denom).to("dB")})

    @property
    def tabular_margin(self) -> OrderedDict:
        """Returns an ordered dictionary of parameters for tabular display of the link margin."""
        numer = {
            "pt": self.pt.to("dBm"),
            "gt": self.gt,
            "gr": self.gr,
            "coding_gain": self.coding_gain,
        }
        denom = {
            "path_loss": self.path_loss,
            "additional_loss": self.additional_loss,
            "k": (1 * ureg.boltzmann_constant).to(ureg.dBm / ureg.kelvin / ureg.Hz),
            "tsys": self.tsys.to("decibelkelvin"),
            "data_rate": self.data_rate.to("decibelhertz"),
            "required_ebno": self.ebno_req,
        }
        return OrderedDict(
            {**numer, **denom, "margin": dictionary_ratio(numer, denom).to("dB")},
        )

    def set_distance_from_margin(self, link_margin: Quantity) -> None:
        """Set the distance to meet a specified link margin."""
        scale = self.link_margin / link_margin
        self.distance = self.distance * np.sqrt(scale)


def dictionary_ratio(numerator: dict, denominator: dict) -> Quantity:
    """
    Calculate ratio from dictionary of numerator and denominator.

    Parameters
    ----------
    numerator : dict
    denominator : dict

    Returns:
    -------
    ratio
    """
    numer = None
    for v in numerator.values():
        v = v.to_base_units()
        if numer is None:
            numer = v
        else:
            numer = numer * v

    denom = None
    for v in denominator.values():
        v = v.to_base_units()
        if denom is None:
            denom = v
        else:
            denom = denom * v

    return numer / denom
