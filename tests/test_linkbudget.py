import numpy as np
import pytest
from xrench.units import ureg

from xcvr.ber import mpsk_ber, mqam_ber, mqam_ebno
from xcvr.linkbudget import LinkBudget


def test_link_budget_closure():
    """Test basic link budget margin calculation."""
    lb = LinkBudget(
        name="TestLink",
        pt=30 * ureg.dBm,  # 1W
        gt=10 * ureg.dB,  # 10x
        gr=10 * ureg.dB,  # 10x
        nf=3 * ureg.dB,
        distance=1000 * ureg.m,  # 1km
        frequency=2.4 * ureg.GHz,
        bandwidth=1 * ureg.MHz,
        data_rate=1 * ureg.Mbps,
        ber_req=1e-5,
        modulation="qam",
        modulation_order=2,  # QPSK
    )

    # Verify that margin is calculated
    margin = lb.link_margin
    assert isinstance(margin, ureg.Quantity)
    assert margin.units == ureg.dB
    assert margin.magnitude > 0  # Should close at 1km


def test_link_budget_distance_solver():
    """Test solver that adjusts distance based on margin."""
    lb = LinkBudget(
        name="SolverTest",
        pt=0 * ureg.dBm,
        gt=0 * ureg.dB,
        gr=0 * ureg.dB,
        nf=5 * ureg.dB,
        distance=100 * ureg.m,
        frequency=1 * ureg.GHz,
        bandwidth=1 * ureg.MHz,
        data_rate=1 * ureg.Mbps,
        ber_req=1e-3,
        modulation="psk",
        modulation_order=1,  # BPSK
    )

    initial_dist = lb.distance.magnitude
    print(lb.link_margin)
    old_margin = lb.link_margin
    increase_margin = 3 * ureg.dB
    lb.set_distance_from_margin(old_margin * increase_margin)
    # If we want more margin, distance must decrease
    assert lb.distance.magnitude < initial_dist
    assert np.isclose((lb.link_margin - old_margin), increase_margin)


def test_ber_functions():
    """Verify BER and Eb/N0 solvers."""
    # Test QPSK (p=2) BER at 10dB Eb/No
    ebno = 10 * ureg.dB
    ber = mqam_ber(p=2, ebno=ebno)
    assert ber < 1e-3

    # Test reverse solver (EbNo for a target BER)
    target_ber = 1e-5
    req_ebno = mqam_ebno(p=2, ber=target_ber)
    # Verify that the calculated EbNo results in roughly the target BER
    calculated_ber = mqam_ber(p=2, ebno=req_ebno)
    assert np.isclose(calculated_ber, target_ber, rtol=0.1)
