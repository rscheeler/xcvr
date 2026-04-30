"""Module for symbols used in block diagrams."""

from typing import Any

import schemdraw
import schemdraw.elements


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


class Attenuator(schemdraw.dsp.Square):
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


class Coupler(schemdraw.dsp.Square):
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
        thru = [(0, 0.4), (1, 0.4)]
        self.segments.append(schemdraw.segments.Segment(thru))
        cpl = [(0, -0.4), (1, -0.4)]
        self.segments.append(schemdraw.segments.Segment(cpl))
        x1 = [(0.15, 0.35), (0.85, -0.35)]
        self.segments.append(schemdraw.segments.Segment(x1))
        x2 = [(0.15, -0.35), (0.85, 0.35)]
        self.segments.append(schemdraw.segments.Segment(x2))


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
