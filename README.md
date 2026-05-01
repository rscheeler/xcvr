# 📻 xcvr

**xcvr** is a Python library for RF transceiver cascaded analysis. Define components from S-parameter data or datasheet constants, cascade them into systems, and analyze gain, noise figure, linearity, compression, and link budgets — all frequency-aware and unit-safe via [pint](https://pint.readthedocs.io/) and [xarray](https://docs.xarray.dev/).

Built on top of [scikit-rf](https://scikit-rf.readthedocs.io/), [schemdraw](https://schemdraw.readthedocs.io/), and [xarray](https://docs.xarray.dev/).

## Installation

```bash
uv add xcvr
```

Or with pip:

```bash
pip install xcvr
```

## Features

- **Cascaded RF system analysis** — gain, noise figure, OIP3, IIP3, P1dB, Psat, and noise temperature across the full device chain, frequency-resolved
- **S-parameter based devices** — load real `.sNp` touchstone files via scikit-rf for accurate, broadband device models
- **Constant (datasheet) devices** — quickly define devices from scalar gain and noise figure values for early-stage budgets
- **Cable modeling** — frequency-dependent coaxial cable loss from physical parameters (length, tan δ)
- **Nested subsystems** — compose `System` objects from other `System` objects; expand or collapse subsystems in analysis and diagrams
- **Link budget analysis** — complete RF link budget with Eb/N0, CNR, link margin, G/T, BER, and max range calculations for M-ary PSK and QAM modulations
- **Signal compression tracking** — propagate signal level through the chain with Psat limiting at each stage
- **Block diagrams** — auto-generated schematic block diagrams via schemdraw with per-device symbols
- **Power dissipation** — track supply voltage, current, and cumulative power dissipation per device
- **Rich DataFrame output** — styled pandas tables with bar charts and compression highlighting
- **Unit safety** — all quantities carry pint units; dB, dBm, kelvin, Hz conversions are handled automatically

## Quick Start

### System Cascade

```python
import skrf
import schemdraw
import xarray as xr
from xrench.units import ureg

from xcvr import Device, System, Symbol, plot_cascade, plot_signal_compression
from xcvr.symbols import Attenuator, Coupler
from xcvr.devices import Cable, Constant

# --- Define devices from S-parameter files ---
lna_net = skrf.Network("SKY55950_WB.s2p")

lna = Device(
    "A1", "Skyworks", "SKY55950",
    network=lna_net,
    nf=2 * ureg.dB,
    p1db=10 * ureg.dBm,
    oip3=-5 * ureg.dBm,
    vsup=2.8 * ureg.volt,
    isup=2 * ureg.mA,
    symbol=Symbol(schemdraw.dsp.Amp),
)

att_net = skrf.Network("GAT-1+.s2p")
attenuator = Device(
    "R1", "Susumu", "PAT0510S-C-1DB",
    network=att_net,
    symbol=Attenuator,
)

# --- Build a system ---
rx = System(
    "GPS RX", "xcvr inc.", "001",
    [lna, attenuator, lna, attenuator],
)

# --- View the block diagram ---
rx.block_diagram()

# --- Cascaded NF at GPS L1 ---
rx.cascaded_nf.sel(frequency=1.575e9, method="nearest")

# --- Full cascade table at a single frequency ---
f = (1.575 * ureg.GHz).to("Hz")
rx.get_dataframe(input_power=0 * ureg.dBm, frequency=f.magnitude, method="nearest")

# --- Plot cascaded gain vs. device ---
ax = plot_cascade(rx, prop="gain", frequency=1.575e9)
ax.grid()

# --- Plot signal level and compression headroom ---
input_pwr = xr.DataArray([-20] * ureg.dBm, dims=("frequency",), coords={"frequency": [1.575e9]})
ax = plot_signal_compression(rx, input_pwr, frequency=1.575e9)
ax.grid()
```

### Constant (Datasheet) Device

Use `Constant` to define a device from a single gain value when S-parameter files are not yet available:

```python
from xcvr.devices import Constant

frequency = xr.DataArray(
    [1.0e9, 1.5e9, 2.0e9] * ureg.Hz, dims=("frequency",), coords={"frequency": [1.0e9, 1.5e9, 2.0e9]}
)

amp = Constant(
    "A1", "Generic", "AMP-001",
    frequency=frequency,
    gain=20 * ureg.dB,
    noise_gain=20 * ureg.dB,
    nf=3 * ureg.dB,
    p1db=15 * ureg.dBm,
)
```

### Cable

```python
from xcvr.devices import Cable

cable = Cable(
    "W1", "Times Microwave", "LMR-400",
    frequency=frequency,
    length=10 * ureg.inch,
    tan_delta=0.002,
)
```

### Nested Subsystems

```python
frontend = System("Frontend", "xcvr", "FE-001", [lna, attenuator])
backend  = System("Backend",  "xcvr", "BE-001", [attenuator, lna, attenuator])

# Compose into a top-level system
full_rx = System("Full RX", "xcvr", "RX-001", [frontend, backend])

# Expand the frontend to show its individual devices in the block diagram
frontend.expand = True
full_rx.block_diagram()
```

### Link Budget

```python
from xcvr.linkbudget import LinkBudget
from xrench.units import ureg

lb = LinkBudget(
    name="UHF Downlink",
    pt=2 * ureg.watt,
    gt=6 * ureg.dB,
    gr=3 * ureg.dB,
    nf=4 * ureg.dB,
    distance=500 * ureg.km,
    frequency=437 * ureg.MHz,
    bandwidth=200 * ureg.kHz,
    data_rate=9600 * ureg.bit / ureg.second,
    ber_req=1e-6,
    modulation="psk",
    modulation_order=2,
)

print(lb.link_margin)          # dB margin
print(lb.ebno)                 # achieved Eb/N0
print(lb.ebno_req)             # required Eb/N0
print(lb.max_link_distance)    # max range in km

# Rendered HTML table in Jupyter
from IPython.display import HTML, display
display(HTML(lb.view_link_table("tabular_margin")))
```

### BER Curves

```python
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from xrench.units import ureg
from xcvr.ber import mqam_ber

ebno = xr.DataArray(
    np.linspace(0, 25, 101) * ureg.dB,
    dims=("ebno",),
    coords={"ebno": np.linspace(0, 25, 101)},
)
mqam_ber(ebno=ebno).plot.line(x="ebno", yscale="log")
plt.gca().set(xlabel="Eb/N0 [dB]", ylabel="BER")
plt.grid(True)
```

## API Reference

### `Device`

The base component. Wraps a `skrf.Network` and optional scalar performance parameters.

| Parameter | Description |
|----------:|:------------|
| `name` | Designator string (e.g. `"A1"`) |
| `manufacturer` | Manufacturer name |
| `pn` | Part number |
| `network` | `skrf.Network` S-parameter data |
| `nf` | Noise figure (`Quantity` or `DataArray`) |
| `p1db` | Output-referred 1 dB compression point |
| `oip3` | Output-referred IP3 |
| `psat` | Output-referred saturation power |
| `vsup` / `isup` | Supply voltage / current for power dissipation |

Key properties: `.gain`, `.nf`, `.p1db`, `.oip3`, `.iip3`, `.psat`, `.noise_temperature`, `.noise_gain`

### `System`

A cascade of `Device` and/or nested `System` objects.

| Method / Property | Description |
|------------------:|:------------|
| `.cascaded_gain` | Cumulative gain at each stage |
| `.cascaded_nf` | Cumulative Friis noise figure |
| `.cascaded_oip3` | Cascaded OIP3 |
| `.cascaded_p1db` | Cascaded P1dB |
| `.cascaded_psat` | Cascaded Psat with limiting |
| `.cascaded_pdiss` | Cumulative power dissipation |
| `.get_signal_level(pin)` | Signal level at each stage with Psat limiting |
| `.get_dataframe(...)` | Styled pandas DataFrame of all parameters |
| `.get_dataset(...)` | xarray Dataset of all parameters |
| `.block_diagram()` | schemdraw block diagram |
| `.network` | Cascaded `skrf.Network` |

### `LinkBudget`

Frozen dataclass representing a complete RF link.

Key properties: `.eirp`, `.path_loss`, `.tsys`, `.g_over_t`, `.cnr`, `.ebno`, `.ebno_req`, `.link_margin`, `.ber`, `.max_path_loss`, `.max_link_distance`

Tabular views: `.tabular_margin`, `.tabular_cnr`, `.tabular_ebno` (all return `OrderedDict`), rendered via `.view_link_table()`

### BER Functions (`xcvr.ber`)

| Function | Description |
|---------:|:------------|
| `mqam_ber(p, ebno)` | M-QAM BER vs. Eb/N0 |
| `mpsk_ber(p, ebno)` | M-PSK BER vs. Eb/N0 |
| `mqam_ebno(p, ber)` | Required Eb/N0 for M-QAM at a target BER |
| `mpsk_ebno(p, ber)` | Required Eb/N0 for M-PSK at a target BER |

Modulation order `p` is the exponent: `m = 2**p` (e.g. `p=2` → 4-QAM/QPSK, `p=3` → 8-PSK).


## Dependencies

- [scikit-rf](https://scikit-rf.readthedocs.io/) — S-parameter network manipulation
- [schemdraw](https://schemdraw.readthedocs.io/) — block diagram drawing
- [xarray](https://docs.xarray.dev/) — labeled N-D arrays
- [pint](https://pint.readthedocs.io/) — unit-aware quantities
- [xrench](https://github.com/rscheeler/xrench) — xarray + pint integration utilities
- numpy, scipy, pandas, matplotlib

## License

MIT

## Author

Created by [Rob Scheeler](https://github.com/rscheeler)