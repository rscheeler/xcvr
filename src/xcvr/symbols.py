"""Module for symbols used in block diagrams."""

from __future__ import annotations

from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from typing import Any

import schemdraw
import schemdraw.elements
from schemdraw import dsp
from schemdraw.segments import Segment, SegmentText
from schemdraw.transform import Transform


class Symbol:
    """
    Symbol container for the schemdraw Element object. The container delays the object instance
    creation until the block diagram is requested.
    """

    def __init__(self, element: schemdraw.elements.Element, *args: Any, **kwargs: Any) -> None:
        self.element = element
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args: Any, **kwargs: Any) -> schemdraw.elements.Element:
        args = list(args) + list(self.args)
        kwargs = {**kwargs, **self.kwargs}
        return self.element(*args, **kwargs)


class Attenuator(dsp.Square):
    """
    Box with a rotated resistor in it.

        Anchors:
        * N
        * S
        * E
        * W
    """

    def __init__(self, *d: Any, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(*d, **kwargs)
        scale = 0.8
        resheight = 0.25 * scale  # Resistor height
        reswidth = 1.0 / 6 * scale  # Full (inner) length of resistor is 1.0 data unit
        margin = 0.25
        path = ((margin, 0), (1 - margin, 0))
        res_segs = [
            (0, 0),
            (0.5 * reswidth, resheight),
            (1.5 * reswidth, -resheight),
            (2.5 * reswidth, resheight),
            (3.5 * reswidth, -resheight),
            (4.5 * reswidth, resheight),
            (5.5 * reswidth, -resheight),
            (6 * reswidth, 0),
        ]
        res_segs = [rs[::-1] for rs in res_segs]
        res_segs = [(rs[0] + 0.5, rs[1] - 0.5 * scale) for rs in res_segs]
        self.segments.append(schemdraw.segments.Segment(res_segs))


class VariableAttenuator(Attenuator):
    """Attenuator with a diagonal arrow through it (variable attenuation).

    Anchors: N, S, E, W
    """

    def __init__(self, *d: Any, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(*d, **kwargs)
        self.segments.append(
            Segment([(0.1, -0.4), (0.9, 0.4)], arrow="->", arrowwidth=0.12, arrowlength=0.18),
        )


class Coupler(dsp.Square):
    """
    Directional coupler.

        Anchors:
        * N
        * S
        * E
        * W
    """

    def __init__(self, *d: Any, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(*d, **kwargs)
        thru = [(0, 0.4), (1, 0.4)]
        self.segments.append(schemdraw.segments.Segment(thru))
        cpl = [(0, -0.4), (1, -0.4)]
        self.segments.append(schemdraw.segments.Segment(cpl))
        x1 = [(0.15, 0.35), (0.85, -0.35)]
        self.segments.append(schemdraw.segments.Segment(x1))
        x2 = [(0.15, -0.35), (0.85, 0.35)]
        self.segments.append(schemdraw.segments.Segment(x2))


class Splitter(dsp.Square):
    """
    Two-way power splitter.

        Anchors:
        * N
        * S
        * E
        * W
    """

    def __init__(self, *d: Any, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(*d, **kwargs)
        start = [(0, 0), (0.35, 0)]
        self.segments.append(schemdraw.segments.Segment(start))
        split_up = [(0.35, 0), (0.85, 0.35)]
        self.segments.append(schemdraw.segments.Segment(split_up))
        split_down = [(0.35, 0), (0.85, -0.35)]
        self.segments.append(schemdraw.segments.Segment(split_down))
        up_end = [(0.85, 0.35), (1, 0.35)]
        self.segments.append(schemdraw.segments.Segment(up_end))
        down_end = [(0.85, -0.35), (1, -0.35)]
        self.segments.append(schemdraw.segments.Segment(down_end))


class Combiner(dsp.Square):
    """
    Two-way power splitter.

        Anchors:
        * N
        * S
        * E
        * W
    """

    def __init__(self, *d: Any, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(*d, **kwargs)
        start = [(0.85, 0), (1, 0)]
        self.segments.append(schemdraw.segments.Segment(start))
        split_up = [(0.85, 0), (0.35, 0.35)]
        self.segments.append(schemdraw.segments.Segment(split_up))
        split_down = [(0.85, 0), (0.35, -0.35)]
        self.segments.append(schemdraw.segments.Segment(split_down))
        up_end = [(0, 0.35), (0.35, 0.35)]
        self.segments.append(schemdraw.segments.Segment(up_end))
        down_end = [(0.0, -0.35), (0.35, -0.35)]
        self.segments.append(schemdraw.segments.Segment(down_end))


class Port(schemdraw.elements.Element):
    """Port."""

    def __init__(self, *d: Any, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(*d, **kwargs)
        lead = 0.6
        h = 0.5
        w = 0.4
        flat = 0.2
        self.segments.append(
            schemdraw.segments.Segment(
                [
                    (0, 0),
                    (0, lead),
                    (-w, lead + h),
                    (-w, lead + h + flat),
                    (w, lead + h + flat),
                    (w, lead + h),
                    (0, lead),
                ],
            ),
        )
        self.params["drop"] = (0, 0)
        self.params["theta"] = 0
        self.anchors["start"] = (0, 0)
        self.anchors["center"] = (0, 0)
        self.anchors["end"] = (0, 0)


class Line(schemdraw.elements.Element2Term):
    """Box like resistor."""

    def __init__(self, *d: Any, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(*d, **kwargs)

        resheight = 0.4  # Resistor height
        reswidth = 1.0 / 4  # Full (inner) length of resistor is 1.0 data unit
        self.segments.append(
            schemdraw.segments.Segment(
                [
                    (0, 0),
                    (0, resheight),
                    (reswidth * 6, resheight),
                    (reswidth * 6, -resheight),
                    (0, -resheight),
                    (0, 0),
                    schemdraw.elements.elements.gap,
                    (reswidth * 6, 0),
                ],
            ),
        )


class PhaseShifter(dsp.Circle):
    """Circle with a diagonal line through it. Pass ``variable=True`` to make the line an arrow.

    Anchors: N, S, E, W (plus the compass points from ``dsp.Circle``)
    """

    def __init__(self, *d: Any, variable: bool = False, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(*d, **kwargs)
        r = 0.5
        reach = r * 0.7071 + 0.12  # line pokes slightly past the circle
        line = [(r - reach, -reach), (r + reach, reach)]
        if variable:
            self.segments.append(Segment(line, arrow="->", arrowwidth=0.12, arrowlength=0.18))
        else:
            self.segments.append(Segment(line))


class SigmaSum(dsp.Circle):
    """Summing junction: circle with a Sigma.

    Anchors: N, S, E, W (plus the compass points from ``dsp.Circle``)
    """

    def __init__(self, *d: Any, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(*d, **kwargs)
        self.segments.append(
            SegmentText((0.5, 0), r"$\Sigma$", fontsize=16, align=("center", "center")),
        )


class DeltaSplit(dsp.Circle):
    """Splitter: circle with a Delta.

    Anchors: N, S, E, W (plus the compass points from ``dsp.Circle``)
    """

    def __init__(self, *d: Any, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(*d, **kwargs)
        self.segments.append(
            SegmentText((0.5, 0), r"$\Delta$", fontsize=16, align=("center", "center")),
        )


@contextmanager
def building_composite() -> Generator[None]:
    """Stop schemdraw from auto-adding elements created inside a composite's ``__init__``.

    Inside ``with schemdraw.Drawing() as d:`` every Element you instantiate is registered with the
    drawing (that is what lets you skip ``d +=``). A composite creates *child* elements while it is
    still being built, so without this guard the children get drawn on their own and the
    half-built composite is placed too early. schemdraw's own compound elements use the same switch.
    """
    try:
        from schemdraw import drawing_stack  # noqa: PLC0415
    except ImportError:  # very old schemdraw: no context-manager auto-add, nothing to guard
        yield
        return
    previous = drawing_stack.pause
    drawing_stack.pause = True
    try:
        yield
    finally:
        drawing_stack.pause = previous


class CompositeElement(schemdraw.elements.Element):
    """Base class for symbols assembled from other symbols.

    Subclasses call :meth:`embed` and :meth:`wire` from within ``with building_composite():``.
    All coordinates are in the composite's local frame, with the ``W`` anchor at (0, 0).
    """

    def embed(
        self,
        child: Callable[[], schemdraw.elements.Element],
        x: float = 0.0,
        y: float = 0.0,
    ) -> float:
        """Instantiate ``child`` (class, Symbol wrapper or callable), copy its segments to (x, y).

        The child's local origin is assumed to be its W/input-centre, which holds for
        ``dsp.Square``/``dsp.Circle``/``dsp.Amp``-style symbols. Returns the child's width.
        """
        elm = child()
        xf = Transform(theta=0, globalshift=(x, y))
        for seg in elm.segments:
            self.segments.append(seg.xform(xf))
        return float(elm.get_bbox(includetext=False).xmax)

    def wire(self, x0: float, x1: float, y: float = 0.0) -> None:
        """Horizontal wire."""
        if x1 > x0:
            self.segments.append(Segment([(x0, y), (x1, y)]))


StageSpec = str | Callable[[], schemdraw.elements.Element]

#: Names usable in ``stages=(...)``. Anything else can be passed as a class / Symbol / callable.
DEFAULT_STAGES: dict[str, Callable[[], schemdraw.elements.Element]] = {
    "phase": PhaseShifter,
    "atten": VariableAttenuator,
    "amp": dsp.Amp,
    "sum": SigmaSum,
    "split": DeltaSplit,
    "vga": dsp.VGA,
}


class Beamformer(CompositeElement):
    """Single-channel beamformer path: [split] -> phase shifter -> attenuator -> amp -> [sum].

    Args:
        stages: Signal chain in signal order. Each entry is a name from ``DEFAULT_STAGES``
            ('phase', 'atten', 'amp', 'sum', 'split') or any callable returning an Element
            (a class, ``functools.partial``, or your ``Symbol(...)`` wrapper).
        mode: ``'sum'`` appends a Sigma at the end of the path, ``'split'`` prepends a Delta at the
            start, ``None`` adds neither (use this if you list 'sum'/'split' in ``stages`` yourself).
        n: If given, labels the sum/split symbol "xN" (e.g. ``n=8`` -> x8).
        title: Text at the top of the boundary box (None for no title).
        boundary: Draw a dashed boundary around the whole block.
        gap: Wire length between adjacent blocks (and from the boundary to the first/last block).

    Anchors:
        W / in, E / out, N, S, center, plus one anchor per stage at its top-centre
        (``phase``, ``atten``, ``amp``, ``sum``, ``split``; callables use their class name,
        lower-cased; duplicates get a numeric suffix). Handy for control lines.
    """

    def __init__(
        self,
        *d: Any,  # noqa: ANN401
        stages: Sequence[StageSpec] = ("phase", "atten", "amp"),
        mode: str | None = "sum",
        n: int | None = None,
        title: str | None = "Beamformer",
        boundary: bool = True,
        gap: float = 0.4,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(*d, **kwargs)
        if mode not in ("sum", "split", None):
            msg = f"mode must be 'sum', 'split' or None, got {mode!r}"
            raise ValueError(msg)
        chain: list[StageSpec] = list(stages)
        if mode == "sum":
            chain.append("sum")
        elif mode == "split":
            chain.insert(0, "split")
        with building_composite():  # children are created below; don't let a Drawing grab them
            self._build(chain, n, title, boundary, gap)

    def _build(
        self,
        chain: Sequence[StageSpec],
        n: int | None,
        title: str | None,
        boundary: bool,
        gap: float,
    ) -> None:
        x = 0.0
        used: dict[str, int] = {}
        for j, spec in enumerate(chain):
            if isinstance(spec, str):
                name, factory = spec, DEFAULT_STAGES[spec]
            else:
                name, factory = getattr(spec, "__name__", f"s{j}").lower(), spec

            self.wire(x, x + gap)
            x += gap
            w = self.embed(factory, x, 0.0)
            used[name] = used.get(name, 0) + 1
            anchor = name if used[name] == 1 else f"{name}{used[name]}"
            self.anchors[anchor] = (x + w / 2, 0.5)
            if n is not None and name in ("sum", "split"):
                self.segments.append(
                    SegmentText(
                        (x + w / 2, -0.62),
                        rf"$\times${n}",
                        fontsize=10,
                        align=("center", "top"),
                    ),
                )
            x += w
        self.wire(x, x + gap)
        total = x + gap

        half_h = 0.5 + 0.35 + (0.3 if title else 0.0)
        if n is not None:
            half_h = max(half_h, 1.1)  # room for the "xN" label
        if boundary:
            self.segments.append(
                Segment(
                    [(0, -half_h), (0, half_h), (total, half_h), (total, -half_h), (0, -half_h)],
                    ls="-",
                ),
            )
        if title:
            self.segments.append(
                SegmentText(
                    (total / 2, half_h - 0.08),
                    title,
                    fontsize=11,
                    align=("center", "top"),
                ),
            )

        self.anchors["W"] = self.anchors["in"] = (0, 0)
        self.anchors["E"] = self.anchors["out"] = (total, 0)
        self.anchors["N"] = (total / 2, half_h)
        self.anchors["S"] = (total / 2, -half_h)
        self.anchors["center"] = (total / 2, 0)
        self.elmparams["drop"] = (total, 0)
        self.elmparams["lblloc"] = "center"
        self.elmparams["lblofst"] = 0
