from typing import List, Union, Optional
from collections import namedtuple

import schemdraw.elements
from . import symbols as sym
import skrf
import schemdraw
from schemdraw import elements as elm
import numpy as np
from pint import Quantity
from . import ureg
from .rfcascade import Device
from scipy.optimize import minimize
from functools import partial
from copy import copy, deepcopy
from operator import itemgetter

Node = namedtuple("Node", ["x", "y"])


def dequantify(func):
    def wrapped(*args, **kwargs):
        cargs = []
        for a in args:
            if isinstance(a, Quantity):
                cargs.append(a.to(ureg.parse_unit_name(str(a.units))[0][1]).magnitude)
            else:
                cargs.append(a)
        ckwargs = dict()
        for k, v in kwargs.items():
            if isinstance(a, Quantity):
                ckwargs[k] = a.to(ureg.parse_unit_name(str(a.units))[0][1]).magnitude
            else:
                ckwargs[k] = a

        return func(*cargs, **ckwargs)

    return wrapped


class CircuitElement(object):
    """
    Element container for skrf media object. The container delays the object instance creation until the block diagram is requested.
    """

    def __init__(
        self,
        media: Union[skrf.media.Media, skrf.circuit.Circuit],
        circuit: str,
        nodes: tuple,
        name,
        *circuit_args,
        symbol_offset=np.array([0, 0]),
        symbol_args=[],
        symbol_kwargs: Optional[dict] = None,
        network=None,
        lblloc="top",
        **circuit_kwargs,
    ):
        if network is None:
            # Parse the args and kwargs
            cargs = []
            for ca in circuit_args:
                if isinstance(ca, Quantity):
                    cargs.append(ca.to(ureg.parse_unit_name(str(ca.units))[0][1]).magnitude)
                else:
                    cargs.append(ca)
            ckwargs = dict()
            for k, v in circuit_kwargs.items():
                if isinstance(v, Quantity):
                    ckwargs[k] = v.to(ureg.parse_unit_name(str(v.units))[0][1]).magnitude
                else:
                    ckwargs[k] = v
            self.network = getattr(media, circuit)(*cargs, name=name, **ckwargs)
        else:
            self.network = network
        self.name = name
        try:
            symbol = getattr(sym, circuit.title())
        except:
            symbol = getattr(elm, circuit.title())
        # Format value string
        carg_fmt = []
        for a in circuit_args:
            if isinstance(a, Quantity):
                carg_fmt.append(f"{a:~P}")
            else:
                carg_fmt.append(f"{a}")
        valstr = "\n".join(carg_fmt)
        ckstr = circuit_kwargs.copy()
        if "frequency" in ckstr.keys():
            ckstr.pop("frequency")
        if "unit" in ckstr.keys():
            ckstr.pop("unit")
        cksl = []
        for k, v in ckstr.items():
            if isinstance(v, Quantity):
                cksl.append(f"{k}={v:~P}")
            else:
                cksl.append(f"{k}={v}")
        # Get media items if present
        mstr = str(media).split("\n")
        if len(mstr) > 1:
            media_txt = "\n" + "\n".join([m.strip() for m in mstr[1].split(",")])
        else:
            media_txt = ""
        if len(cksl) > 0:
            valstr += "\n" + "\n".join(cksl)

        label = name + "\n" + valstr + media_txt
        lblkwargs = dict(label=label, lblloc=lblloc)

        # Initialize kwargs
        if symbol_kwargs is None:
            symbol_kwargs = dict()
        symbol_kwargs = {**lblkwargs, **symbol_kwargs}
        self.symbol = sym.Symbol(symbol, *symbol_args, **symbol_kwargs)
        self.nodes = nodes
        self.symbol_offset = symbol_offset


class Circuit:
    def __init__(self, nodes, elems: List[CircuitElement], name="", manufacturer="", pn=""):
        self.elems = elems
        self.nodes = nodes
        self._connections = None
        self._circuit = None
        self._network = None
        self.name = name
        self.manufacturer = manufacturer
        self.pn = pn

    @property
    def connections(self):
        if self._connections is None:
            _connections = []
            for node in range(len(self.nodes)):
                node_conns = []
                for e in self.elems:
                    if node in e.nodes:
                        node_conns.append((e.network, e.nodes.index(node)))
                _connections.append(node_conns)
            self._connections = _connections
        return self._connections

    @property
    def circuit(self):
        if self._circuit is None:
            self._circuit = skrf.Circuit(self.connections)
            self._circuit.network.name = self.name
        return self._circuit

    @property
    def network(self):
        if self._network is None:
            self._network = self.circuit.network
            self._network.name = self.name
        return self._network

    def element(self, show_ports=True, show_nodes=True) -> elm.Element:
        elm.ElementCompound

    def symbol(self, show_ports=True, show_nodes=True):
        dwg = _CircuitCE(self, show_ports=show_ports, show_nodes=show_nodes)

        return dwg

    def as_device(self, **kwargs):
        """
        Convert circuit to device
        """
        dev = Device(
            self.name,
            self.manufacturer,
            self.pn,
            self.network,
            symbol=sym.Symbol(self.symbol, show_ports=False, show_nodes=False),
            **kwargs,
        )

        return dev


def connect_circuits(
    c0: Circuit,
    p0: int,
    c1: Circuit,
    p1: int,
    name: str = "",
    manufacturer: str = "",
    pn: str = "",
):
    """
    Connect circuit 0 and circuit 1.
    """
    c0c = deepcopy(c0)
    c1c = deepcopy(c1)

    # Get Circuit Elments
    for e in c0c.elems:
        if e.name == p0:
            p0e = e

    for e in c1c.elems:
        if e.name == p1:
            p1e = e

    # Get index of node to be popped
    p1ni = p1e.nodes[0]
    p0ni = p0e.nodes[0]

    # Remove Node on Circuit 1
    c1c.nodes.pop(p1e.nodes[0])

    # Remove Ports
    c0c.elems.pop(c0c.elems.index(p0e))
    c1c.elems.pop(c1c.elems.index(p1e))

    for i, n in enumerate(c1c.nodes):
        # Update both the Circuit
        newn = n._replace(x=n.x + c0c.nodes[p0e.nodes[0]].x, y=n.y + c0c.nodes[p0e.nodes[0]].y)
        c1c.nodes[i] = newn

    # Shift indexes or replace
    for el in c1c.elems:
        nshift = list(el.nodes)
        for ii, eln in enumerate(el.nodes):
            if eln == p1ni:
                nshift[ii] = p0ni
            else:
                nshift[ii] = eln + len(c0c.nodes) - 1
        el.nodes = nshift

    # Combine nodes
    nodes = c0c.nodes + c1c.nodes
    # Combine elements
    elems = c0c.elems + c1c.elems

    new_circ = Circuit(nodes, elems, name=name, manufacturer=manufacturer, pn=pn)

    return new_circ


class _CircuitCE(elm.ElementCompound):
    def __init__(self, parent, show_ports=True, show_nodes=True):
        self.show_ports = show_ports
        self.show_nodes = show_nodes
        self.parent = parent
        super().__init__()

    def setup(self, show_ports=True, show_nodes=True):
        shift = [[1, 0], [0, 1]]

        # Place nodes
        if show_nodes:
            for i, n in enumerate(self.parent.nodes):
                self.add(elm.Dot(xy=n).label(f"{i}", ofst=(0.3, 0.1)))

        locs = []
        for e in self.parent.elems:
            ofst = e.symbol_offset.copy()
            if len(e.nodes) == 2:
                # Determine if parallel element
                parallel = False
                if len(locs) != 0:
                    node_intersection = np.array(locs) - np.array(itemgetter(*e.nodes)(self.parent.nodes))
                    for i in range(node_intersection.shape[0]):
                        ni = node_intersection[i, ...]
                        if np.count_nonzero(ni == 0) > 2:
                            parallel = True
                if parallel:
                    diff = np.diff(np.array(itemgetter(*e.nodes)(self.parent.nodes)), axis=0).squeeze().tolist()
                    shift_idx = diff.index(0)
                    ofst += shift[shift_idx]
                if isinstance(e.symbol(), schemdraw.dsp.Square):
                    # Square should be 1 unit wide
                    n0 = np.array(self.parent.nodes[e.nodes[0]])
                    n1 = np.array(self.parent.nodes[e.nodes[1]])
                    diff = n1 - n0
                    a = "1"

                    mid = n0 + diff / 2 - np.array([0.5, 0])
                    l = elm.Line(at=n0, to=mid)
                    self.add(l)
                    s = e.symbol()
                    self.add(s)
                    l = elm.Line(at=s.E, to=n1)
                    self.add(l)
                else:
                    self.add(
                        e.symbol(
                            at=np.array(self.parent.nodes[e.nodes[0]]) + ofst,
                            to=np.array(self.parent.nodes[e.nodes[1]]) + ofst,
                        )
                    )
                # Add connecting lines
                self.add(
                    elm.Line(
                        at=np.array(self.parent.nodes[e.nodes[0]]),
                        to=np.array(self.parent.nodes[e.nodes[0]]) + ofst,
                    )
                )
                self.add(
                    elm.Line(
                        at=np.array(self.parent.nodes[e.nodes[1]]),
                        to=np.array(self.parent.nodes[e.nodes[1]]) + ofst,
                    )
                )
                locs.append(itemgetter(*e.nodes)(self.parent.nodes))

            else:
                if e.symbol.element == sym.Port and not show_ports:
                    pass
                else:
                    self.add(e.symbol(at=np.array(self.parent.nodes[e.nodes[0]]) + ofst))

                    self.add(
                        elm.Line(
                            at=np.array(self.parent.nodes[e.nodes[0]]),
                            to=np.array(self.parent.nodes[e.nodes[0]]) + ofst,
                        )
                    )


def lnet_circuit(*d, tlin=None, yymtype=1):
    """
    Generates an L-network Circuit. Types from Yin and Yang of matching part 1 definced by yymtype.
    """

    unit = 4
    nodes = [Node(0, 0), Node(unit, 0), Node(0, -unit)]

    P0 = CircuitElement(
        skrf.Circuit,
        "Port",
        (0,),
        "P0",
        frequency=tlin.frequency,
        z0=50 * ureg.ohm,
        symbol_kwargs=dict(d="up"),
    )
    P1 = CircuitElement(
        skrf.Circuit,
        "Port",
        (1,),
        "P1",
        frequency=tlin.frequency,
        z0=50 * ureg.ohm,
        symbol_kwargs=dict(d="down", lblloc="bot"),
    )
    G = CircuitElement(
        skrf.Circuit,
        "Ground",
        (2,),
        "GND0",
        frequency=tlin.frequency,
        z0=50 * ureg.ohm,
        symbol_kwargs=dict(lblloc="bot"),
    )

    if yymtype == 1:
        D0 = CircuitElement(tlin, "inductor", (0, 1), "L1", d[0] * ureg.nH)
        D1 = CircuitElement(
            tlin,
            "capacitor",
            (1, 2),
            "C1",
            d[1] * ureg.pF,
        )
        nodes[2] = Node(unit, -unit)
    elif yymtype == 2:
        D0 = CircuitElement(
            tlin,
            "capacitor",
            (0, 1),
            "C1",
            d[0] * ureg.pF,
        )
        D1 = CircuitElement(tlin, "inductor", (1, 2), "L1", d[1] * ureg.nH)
        nodes[2] = Node(unit, -unit)

    elif yymtype == 3:
        D0 = CircuitElement(
            tlin,
            "capacitor",
            (0, 2),
            "C1",
            d[0] * ureg.pF,
        )
        D1 = CircuitElement(tlin, "inductor", (0, 1), "L1", d[1] * ureg.nH)

    elif yymtype == 4:
        D0 = CircuitElement(
            tlin,
            "inductor",
            (0, 2),
            "L1",
            d[0] * ureg.nH,
        )
        D1 = CircuitElement(tlin, "capacitor", (0, 1), "C1", d[1] * ureg.pF)

    elif yymtype == 5:
        D0 = CircuitElement(
            tlin,
            "capacitor",
            (0, 1),
            "C1",
            d[0] * ureg.pF,
        )
        D1 = CircuitElement(
            tlin,
            "capacitor",
            (1, 2),
            "C2",
            d[1] * ureg.pF,
        )
        nodes[2] = Node(unit, -unit)

    elif yymtype == 6:
        D0 = CircuitElement(
            tlin,
            "capacitor",
            (0, 2),
            "C1",
            d[0] * ureg.pF,
        )
        D1 = CircuitElement(
            tlin,
            "capacitor",
            (0, 1),
            "C2",
            d[1] * ureg.pF,
        )

    elif yymtype == 7:
        D0 = CircuitElement(tlin, "inductor", (0, 1), "L1", d[0] * ureg.nH)
        D1 = CircuitElement(tlin, "inductor", (1, 2), "L2", d[1] * ureg.nH)
        nodes[2] = Node(unit, -unit)

    elif yymtype == 8:
        D0 = CircuitElement(
            tlin,
            "inductor",
            (0, 2),
            "L1",
            d[0] * ureg.nH,
        )
        D1 = CircuitElement(tlin, "inductor", (0, 1), "L2", d[1] * ureg.nH)
    elif yymtype == 9:
        nodes.pop(2)
        nodes.append(Node(unit * 2, 0))
        R0I = CircuitElement(tlin, "inductor", (nodes[0], nodes[1]), "L1", d[0] * ureg.nH)
        R0C = CircuitElement(tlin, "capacitor", (nodes[0], nodes[1]), "C1", d[1] * ureg.pF)
        R1I = CircuitElement(tlin, "inductor", (nodes[1], nodes[2]), "L2", d[2] * ureg.nH)
        R1C = CircuitElement(tlin, "capacitor", (nodes[1], nodes[2]), "C2", d[3] * ureg.pF)
        R2I = CircuitElement(tlin, "inductor", (nodes[1], nodes[3]), "L3", d[4] * ureg.nH)
        R2C = CircuitElement(tlin, "capacitor", (nodes[1], nodes[3]), "C3", d[5] * ureg.pF)

    match = Circuit(nodes=nodes, elems=[D0, D1, P0, P1, G], name=f"Type {yymtype} Match")
    return match


def optimal_match_worker(x, f0str="1GHz", tlin=None, yymtype=1, load=None):
    """
    Matching worker function.
    """
    _ntw = lnet_circuit(*x, tlin=tlin, yymtype=yymtype).network ** load

    return np.abs(_ntw[f0str].s00.s).ravel()


def optimize_match(load: Circuit, f0str: str, yymtype: int, igs=None, bounds=None):
    """
    Generate a matching circuit of type yymtype from the Yin and Yang of matching part 1.
    Run optimized to match the load at the specified frequency.
    """
    line = skrf.DefinedGammaZ0(frequency=load.network.frequency, z0=50)
    xlen = [None, 2, 2, 2, 2, 2, 2, 2, 2, 6, 6][yymtype]

    # Inital guesses
    ig = 1  # (nH or pF )
    if igs is None:
        igs = [copy(ig) for i in range(xlen)]

    # bounds
    bound = (0.1, 100)  # nH or pF
    if bounds is None:
        bounds = [copy(bound) for i in range(xlen)]

    # Run the optimizer
    res = minimize(
        partial(
            optimal_match_worker,
            f0str=f0str,
            tlin=line,
            yymtype=yymtype,
            load=load.network,
        ),
        igs,
        bounds=bounds,
    )
    # Round values
    vals = np.around(np.array(res.x), decimals=1)

    # Create Circuit
    lmc = lnet_circuit(*vals, tlin=line, yymtype=yymtype)
    matched = connect_circuits(lmc, "P1", deepcopy(load), "P0")

    return matched, lmc
