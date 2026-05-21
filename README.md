# Digital Linearizer Based on 1-Bit Quantizations

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deijany/DSP2023-2023117048_ICCT2024-KT2153/blob/main/linearizer_ICCT_2024.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code for the following publications:

> **[DSP 2023]** D. Rodriguez Linares and H. Johansson, "Low-Complexity Memoryless Linearizer for Analog-to-Digital Interfaces," *24th International Conference on Digital Signal Processing (DSP)*, 2023.
> [[IEEE]](https://ieeexplore.ieee.org/document/10167765/) · [[arXiv]](https://arxiv.org/abs/2304.05849)

> **[ICCT 2024]** D. Rodriguez Linares and H. Johansson, "Digital Linearizer Based on 1-Bit Quantizations," *24th IEEE International Conference on Communication Technology (ICCT)*, 2024.
> [[IEEE]](https://ieeexplore.ieee.org/document/10946352/) · [[arXiv]](https://arxiv.org/abs/2503.02729)

---

## Overview

Analog frontends — ADCs, amplifiers, filters, and mixers — introduce nonlinear distortion that degrades signal quality. This work presents a **digital post-distortion linearizer** that suppresses that distortion without any modification to the analog hardware.

The linearizer reconstructs the clean signal $\hat{x}(n)$ from the distorted observation $v(n)$ using a bank of parallel branches:

$$\hat{x}(n) = c_0 + c_1 v(n) + \sum_{m=1}^{N} w_m \, f_m(v(n) + b_m)$$

Weights $w_m$ are solved in closed form via **regularized matrix inversion** (no gradient-based training needed). The offset grid $b_m$ is selected by a **minimax linesearch** that optimizes worst-case SNR across all training signals.

### Key innovation — 1-bit activations (ICCT 2024)

When $f_m$ is the 1-bit quantizer (Q1b), the output is binary. Combined with uniformly spaced offsets, the entire linearizer reduces to a **look-up table**, eliminating all multiplications at inference time.

---

## Activation functions

| Name | Description |
|------|-------------|
| `Q1b` | 1-bit quantization — reduces to a look-up table (ICCT 2024 focus) |
| `ReLU` | Rectified linear unit |
| `ABS` | Absolute value |
| `linear` | Linear branches (Hammerstein-like) |
| `polynomial` | Classical Hammerstein structure ($b_m = 0$, fixed offsets) |

---

## Repository structure

```
├── linearizer_ICCT_2024.ipynb   # Main notebook (run this)
├── linearizer_ICCT_2024.py      # Script version
├── classes/
│   ├── linearizers_classes.py   # MatrixInversionLinearizer, ActivationFunctions
│   ├── generator_classes_v4.py  # Multisine signal generator, polynomial distortion model
│   ├── signal_processing.py     # SNR, Statistics
│   └── file_manipulation.py     # Dataset I/O (HDF5), PathManager
├── myfunctions/
│   └── functions_helper.py      # SpectrumAnalyzer, plotting utilities
├── datasets/                    # Pre-generated HDF5 datasets (train/test)
├── trained_model/               # Saved simulation results (.pickle)
└── plots/                       # Output spectrum figures
```

---

## Quickstart

### Run in Google Colab

Click the **Open in Colab** badge above. The setup cell at the top of the notebook handles everything automatically.

### Run locally

**Requirements:** Python 3.9+, numpy, scipy, matplotlib, h5py

```bash
git clone https://github.com/deijany/DSP2023-2023117048_ICCT2024-KT2153.git
cd DSP2023-2023117048_ICCT2024-KT2153
pip install numpy scipy matplotlib h5py
jupyter notebook linearizer_ICCT_2024.ipynb
```

### Key parameters (Section 2 of the notebook)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `activation_function` | `ReLU` | Change to `Q1b`, `ABS`, `linear`, or `polynomial` |
| `branch_number` | 7 | Number of parallel branches |
| `output_resolution` | 12 | Output quantizer resolution (bits) |
| `L2` | 1e-10 | Ridge regularization |
| `number_simulations` | 10 | Linesearch candidates for offset grid |
| `storage_dataset` | `False` | Set `True` to regenerate dataset from scratch |

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{rodriguez2023low,
  title     = {Low-Complexity Memoryless Linearizer for Analog-to-Digital Interfaces},
  author    = {Rodriguez Linares, D. and Johansson, H.},
  booktitle = {24th International Conference on Digital Signal Processing (DSP)},
  year      = {2023}
}

@inproceedings{rodriguez2024digital,
  title     = {Digital Linearizer Based on 1-Bit Quantizations},
  author    = {Rodriguez Linares, D. and Johansson, H.},
  booktitle = {24th IEEE International Conference on Communication Technology (ICCT)},
  year      = {2024}
}
```

---

## License

MIT — see [LICENSE](LICENSE).