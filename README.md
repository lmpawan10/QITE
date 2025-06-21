# QITE
[![arXiv](https://img.shields.io/badge/arXiv-2504.18156-B31B1B.svg)](https://arxiv.org/abs/2504.18156)

This repository contains research code exploring **Quantum Imaginary Time Evolution (QITE)** using [Qiskit](https://qiskit.org/). It accompanies the paper ["Determining Molecular Ground State with Quantum Imaginary Time Evolution using Broken-Symmetry Wave Function"](https://arxiv.org/abs/2504.18156) which demonstrates the technique on a classical emulator. The code here extends those results with support for simulations on the `AerSimulator` and execution on IBM Quantum hardware. It includes scripts for running the algorithm on molecular Hamiltonians and utilities for working with OpenFermion data.

## Repository layout

- `src/` – main modules implementing QITE algorithms and helper utilities to run jobs on IBM Quantum backends.
- `utilities/` – functions for building Hamiltonians and preparing data.
- `examples/` – example scripts for specific molecules (H₄, N₂, P₄) demonstrating usage of the code.
- `data/` – sample integral files used by the example scripts.
- `figures/` – pre-generated plots from previous runs.

### Directory tree

```
QITE/
├── README.md
├── data/
│   ├── 4H_Cluster_Integrals/
│   ├── Ethylene_STO-6G_Integrals/
│   ├── H2_STO-6G_Integrals/
│   ├── Integrals_4H_RHF_UHF/
│   ├── Integrals_N2_BS2/
│   ├── Integrals_N2_BS3/
│   ├── Integrals_N2_RHF/
│   ├── h2.dat
│   ├── h2_bkt_s2.dat
│   └── h2_s2.dat
├── examples/
│   ├── h4/
│   │   └── qiskit_qite_h4_sv.py
│   ├── n2/
│   │   └── qite_matrix_exponential.py
│   └── p4/
│       └── qiskit_qite_p4_sv.py
├── figures/
├── src/
│   ├── get_results.py
│   ├── inputs.py
│   ├── qiskit_qite.py
│   ├── qite_batched_estimator.py
│   ├── real_device.py
│   └── test3_qctrl.py
└── utilities/
    ├── 4h_utility.py
    ├── P4_utility.py
    ├── SparsePauliOp_hamiltonian.py
    ├── hamiltonian.py
    └── prepare_hamiltonian_terms_coeffs.py
```

## Requirements

Install the Python dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Not every package listed there is required for the simplest demonstrations, but installing them all avoids missing-module errors.

Some paths and API tokens are currently hard-coded in `src/inputs.py`. Edit that file or set your own configuration before running the examples.

## Getting started

1. Clone the repository and enter it:

```bash
git clone https://github.com/lmpawan10/QITE.git
cd QITE
```

2. *(Recommended)* Create a Python virtual environment and activate it:

**Linux/macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows**

```cmd
py -m venv .venv
.\.venv\Scripts\activate
```

3. Install the Python dependencies from `requirements.txt`:

**Linux/macOS**

```bash
pip install -r requirements.txt
```

**Windows**

```cmd
py -m pip install -r requirements.txt
```

Not every package in the file is necessary for basic usage, but installing them all ensures nothing is missing. Afterwards you can move into the `src` directory and run the example scripts:

```bash
cd src
```

## Usage

Example workflows are provided under the `examples/` directory. For instance, to run the QITE simulation for N₂ via matrix exponentials:

```bash
python examples/n2/qite_matrix_exponential.py
```

Scripts may require customizing the backend settings and file paths in `src/inputs.py`.

## Citation

If you build on this repository, please cite the companion paper and this code as follows.

**Companion paper**

P. S. Poudel, K. Sugisaki, M. Hajdušek, and R. Van Meter, "Determining Molecular Ground State with Quantum Imaginary Time Evolution using Broken-Symmetry Wave Function," *arXiv*, 2025. Available: <https://arxiv.org/abs/2504.18156>

**Code repository**

P. S. Poudel, K. Sugisaki, M. Hajdušek, and R. Van Meter, *QITE Simulation Code*, GitHub repository, 2025. [Online]. Available: <https://github.com/lmpawan10/QITE>

**BibTeX**

```bibtex
@misc{pawan2025qite,
  title={Determining Molecular Ground State with Quantum Imaginary Time Evolution using Broken-Symmetry Wave Function},
  author={Pawan Sharma Poudel and Kenji Sugisaki and Michal Hajdušek and Rodney Van Meter},
  year={2025},
  eprint={2504.18156},
  archivePrefix={arXiv},
  primaryClass={physics.chem-ph},
  url={https://arxiv.org/abs/2504.18156},
}
```

## References

- *Nature Physics* article, 2019. [Online]. Available: <https://www.nature.com/articles/s41567-019-0704-4>
- *Communications Chemistry* article, 2022. [Online]. Available: <https://www.nature.com/articles/s42004-022-00701-8>

## Contributing

Contributions are welcome! Please fork the repository and open a pull request.

## License

This project is licensed under the [MIT License](LICENSE).