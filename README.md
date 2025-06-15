# QITE
[![arXiv](https://img.shields.io/badge/arXiv-2504.18156-B31B1B.svg)](https://arxiv.org/abs/2504.18156)

This repository contains research code exploring **Quantum Imaginary Time Evolution (QITE)** using [Qiskit](https://qiskit.org/). It accompanies the paper ["Quantum Imaginary Time Evolution"](https://arxiv.org/abs/2504.18156) which demonstrates the technique on a classical emulator. The code here extends those results with support for simulations on the `AerSimulator` and execution on IBM Quantum hardware. It includes scripts for running the algorithm on molecular Hamiltonians and utilities for working with OpenFermion data.

This work also cites [a related Nature Physics study](https://www.nature.com/articles/s41567-019-0704-4) and a follow-up in *Communications Chemistry* (https://www.nature.com/articles/s42004-022-00701-8).

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
│   │   └── qiskit_qite_h4.py
│   ├── n2/
│   │   └── qite_matrix_exponential.py
│   └── p4/
│       └── qiskit_qite_p4.py
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

Install the main Python packages used for QITE:

```bash
pip install qiskit qiskit-aer qiskit-ibmq-provider openfermion numpy scipy matplotlib
```

Not every package is needed for the simplest demonstrations, but installing them avoids missing-module errors. If you keep your own `requirements.txt`, you can install from it instead.

Some paths and API tokens are currently hard-coded in `src/inputs.py`. Edit that file or set your own configuration before running the examples.

## Getting started

1. Clone the repository:

```bash
git clone https://github.com/yourusername/QITE.git
cd QITE/src
```

2. Install the necessary Python packages:

**Linux/macOS**

```bash
pip install qiskit qiskit-aer qiskit-ibmq-provider openfermion numpy scipy matplotlib
```

**Windows**

```cmd
py -m pip install qiskit qiskit-aer qiskit-ibmq-provider openfermion numpy scipy matplotlib
```

After installation you can run the example scripts from this directory. Not every package is required for simple tests, but installing them all avoids missing-module errors.

## Usage

Example workflows are provided under the `examples/` directory. For instance, to run the QITE simulation for N₂ via matrix exponentials:

```bash
python examples/n2/qite_matrix_exponential.py
```

Scripts may require customizing the backend settings and file paths in `src/inputs.py`.

## Citation

If you use this repository or refer to the associated research, please cite it alongside the companion paper:

1. H. Ariff, *"Quantum Imaginary Time Evolution,"* arXiv:2504.18156, 2025.
2. H. Ariff, *QITE Simulation Code*, GitHub repository, 2025. [Online]. Available: https://github.com/yourusername/QITE
3. *Nature Physics* article, 2019. [Online]. Available: https://www.nature.com/articles/s41567-019-0704-4
4. *Communications Chemistry* article, 2022. [Online]. Available: https://www.nature.com/articles/s42004-022-00701-8

## Contributing

Contributions are welcome! Please fork the repository and open a pull request.

## License

This project is licensed under the [MIT License](LICENSE).