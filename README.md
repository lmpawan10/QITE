# QITE

This repository contains research code exploring **Quantum Imaginary Time Evolution (QITE)** using [Qiskit](https://qiskit.org/). It includes scripts for running the algorithm on molecular Hamiltonians and utilities for working with OpenFermion data.

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

The code relies on Python packages such as `qiskit`, `openfermion`, `numpy`, `scipy` and `matplotlib`. Running the examples also requires access to IBM Quantum services with valid credentials.

Some paths and API tokens are currently hard-coded in `src/inputs.py`. Edit that file or set your own configuration before running the examples.

## Usage

Example workflows are provided under the `examples/` directory. For instance, to run the QITE simulation for N₂ via matrix exponentials:

```bash
python examples/n2/qite_matrix_exponential.py
```

Scripts may require customizing the backend settings and file paths in `src/inputs.py`.

## Notes

The repository does not contain an installation script or explicit licensing information. Use at your own discretion and adjust the configuration for your environment.
