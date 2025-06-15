import openfermion
import numpy as np
import sys,shutil
from ..src.inputs import get_h2_s2_integral_files_path, get_4h_integral_files_path

from openfermion.ops import *
from openfermion.transforms import *
from openfermion.linalg import *
from numpy import dot, conjugate, zeros


"""
Generate Hamiltonian and S^2 operator
"""

# Simulation parameters
nso = 8  # Number of spin-orbitals = 4/8/8(H2/P4, 4H/C2H4)
ne  = 4  # Number of electrons = 2/4/4(H2/P4, 4H/C2H4)
mapping_method    = "SCBKT" # JWT, BKT or SCBKT
s_squared_scaling = 1   # s^2 scaling

# For h2
# integral_file_template = get_h2_s2_integral_files_path()
# distance_list = list(map(str, range(120, 290, 10)))
# integral_files = [integral_file_template.replace("H2_R", f"H2_R{num}") for num in distance_list]

# For 4h
integral_file = get_4h_integral_files_path()

#---------- FUNCTION GET_INTEGRALS ----------#
def get_hamiltonian(filename):
    hamiltonian = FermionOperator()
    for line in open(filename):
        items = line.split()
        if len(items) == 1:
            # Nuclear repulsion term
            intval = float(items[0])
            hamiltonian += FermionOperator((), intval)
        elif len(items) == 3:
            # One-electron terms
            iocc = int(items[0])
            avir = int(items[1])
            intval = float(items[2])
            hamiltonian += FermionOperator(((avir, 1), (iocc, 0)), intval)
        elif len(items) == 5:
            # Two-electron terms
            iocc = int(items[0])
            jocc = int(items[1])
            bvir = int(items[2])
            avir = int(items[3])
            intval = float(items[4])
            hamiltonian += FermionOperator(((avir, 1), (bvir, 1), (jocc, 0), (iocc, 0)), intval)
    return hamiltonian


#---------- FUNCTION S_SQUARED_FERMION_DM ----------#
def get_s2_operator(nso, s_squared_scaling):
    # generate S^2 Fermionic operator in DM.
    """
    Notes:
    S(i,j)^2 = S_z(i)*S_z(j) + (S_+(i) * S_-(j) + S_-(i) * S_+(j))/2
    """
    nmo = int(nso // 2)
    s_squared_operator = FermionOperator()
     
    for iorb in range(nmo):
        ia = 2 * iorb
        ib  = 2 * iorb + 1
        for jorb in range(nmo):
            ja = 2 * jorb
            jb  = 2 * jorb + 1
            # S_z(i) * S_z(j) terms
            s_squared_operator +=  0.25 * FermionOperator(((ia, 1), (ia, 0), (ja, 1), (ja, 0)))
            s_squared_operator += -0.25 * FermionOperator(((ia, 1), (ia, 0), (jb, 1), (jb, 0)))
            s_squared_operator += -0.25 * FermionOperator(((ib, 1), (ib, 0), (ja, 1), (ja, 0)))
            s_squared_operator +=  0.25 * FermionOperator(((ib, 1), (ib, 0), (jb, 1), (jb, 0)))
            # (S_+(i) * S_-(j) + S_-(i) * S_+(j))/2 terms
            s_squared_operator +=  0.50 * FermionOperator(((ia, 1), (ib, 0), (jb, 1), (ja, 0)))
            s_squared_operator +=  0.50 * FermionOperator(((ib, 1), (ia, 0), (ja, 1), (jb, 0)))    
    return s_squared_operator * s_squared_scaling

def print_wf(wf, nq):
    wfdim = len(wf)
    thresh = 0.001
    for idet in range(wfdim):
        det = wf[idet]
        if dot(det, conjugate(det)).real >= thresh:
            idet_bin = '{:050b}'.format(idet)
            print(' {:.6f}'.format(det),  '| {}'.format(idet_bin[(50-nq):]),'>')

def extract_hamiltonian_string(hamiltonian_instance):
    """
    Convert a custom Hamiltonian class instance to a string representation.
    
    Args:
        hamiltonian_instance (object): Instance of a Hamiltonian class.
    
    Returns:
        str: String representation of the Hamiltonian.
    """
    if hasattr(hamiltonian_instance, "to_string"):
        return hamiltonian_instance.to_string()
    elif hasattr(hamiltonian_instance, "__str__"):
        return str(hamiltonian_instance)
    else:
        raise TypeError("The Hamiltonian instance cannot be converted to a string.")

def parse_hamiltonian_to_dict(hamiltonian):
    """
    Parse a BK-transformed Hamiltonian into a dictionary.

    Args:
        hamiltonian (str): Hamiltonian in the input format.

    Returns:
        dict: Dictionary with operator strings as keys and coefficients as values.
    """
    ham_dict = {}
    for line in hamiltonian.strip().split("\n"):
        if not line.strip():
            continue  # Skip empty lines

        try:
            
            coeff, ops = line.split(" [", 1)
            ops = ops.rstrip("]")
            coeff = complex(coeff.strip("()").split("+")[0])  # Extract real coefficient
            ham_dict[ops] = ham_dict.get(ops, 0) + coeff
        except ValueError:
            print(f"Warning: Skipping malformed line: {line}")
            continue

    ham_dict = {key.replace('] +', '').strip(): value for key, value in ham_dict.items()}

    return ham_dict

def combine_hamiltonians(ham1_dict, ham2_dict):
    """
    Combine two Hamiltonians given as dictionaries.
    
    Args:
        ham1_dict, ham2_dict (dict): Hamiltonians as dictionaries.
    
    Returns:
        dict: Combined Hamiltonian.
    """
    combined = ham1_dict.copy()
    for key, value in ham2_dict.items():
        combined[key] = combined.get(key, 0) + value
    return combined

def format_hamiltonian_from_dict(ham_dict):
    """
    Format a Hamiltonian dictionary back into the input format.
    
    Args:
        ham_dict (dict): Hamiltonian dictionary.
    
    Returns:
        str: Hamiltonian as a formatted string.
    """
    formatted_terms = []
    for key, coeff in ham_dict.items():
        key = key.strip()  # Ensure no redundant spaces
        if key:  # If the key is not empty
            formatted_terms.append(f"({coeff.real:.12g}+0j) [{key}]")
        else:  # Handle empty terms (identity operator)
            formatted_terms.append(f"({coeff.real:.12g}+0j) []")
    return " +\n".join(formatted_terms).replace("+]", "")


#==========================================================================

# for integral_file in integral_files:
#     if integral_file[103:106] == 'RHF':
#         bs = False
#     else:
#         bs = True
    
# Generate Hamiltonian and S^2 operator
hamiltonian = get_hamiltonian(integral_file)
s2_operator = get_s2_operator(nso, s_squared_scaling)
# print(' Second quantized Hamiltonian')
# print(hamiltonian)
# print('\n Second quantized S^2 operator')
# print(s2_operator)
# Apply fermion-qubit transformation
if mapping_method == 'JWT':
    nq = nso
    qubit_hamiltonian = jordan_wigner(hamiltonian)
    s2_qubit_op = jordan_wigner(s2_operator)
elif mapping_method == 'BKT':
    nq = nso
    qubit_hamiltonian = bravyi_kitaev(hamiltonian)
    s2_qubit_op = bravyi_kitaev(s2_operator)
elif mapping_method == 'SCBKT':
    nq = nso - 2 # In SCBKT we can reduce 2 qubits
    qubit_hamiltonian = symmetry_conserving_bravyi_kitaev(hamiltonian, nso, ne)
    s2_qubit_op = symmetry_conserving_bravyi_kitaev(s2_operator, nso, ne)
qubit_hamiltonian.compress()
s2_qubit_op.compress()
print('\n Qubit Hamiltonian in {}'.format(mapping_method))
print(qubit_hamiltonian)
print('\n Qubit S^2 operator in {}'.format(mapping_method))
print(s2_qubit_op)
qubit_hamiltonian_string = extract_hamiltonian_string(qubit_hamiltonian)
s2_qubit_op_string = extract_hamiltonian_string(s2_qubit_op)
print(f"H String: {qubit_hamiltonian_string}")
qubit_hamiltonian_dict = parse_hamiltonian_to_dict(qubit_hamiltonian_string)
s2_qubit_op_dict = parse_hamiltonian_to_dict(s2_qubit_op_string)
# qubit_hamiltonian_dict = {key.replace('] +', '').strip(): value for key, value in qubit_hamiltonian_dict.items()}
# Motta's H2_Hamiltonian_dict
# h_dict = {'': (-0.3104+0j), 'Z0': (0.1026+0j), 'Z1': (0.0632+0j), 'Z0 Z1': (0.3406+0j), 'X0 X1': (0.1450+0j), 'Y0 Y1': (0.1450+0j)}
# print(f"H Dict: {h_dict}" )
combined_hamiltonian_dict = combine_hamiltonians(qubit_hamiltonian_dict, s2_qubit_op_dict)
combined_hamiltonian_final = format_hamiltonian_from_dict(qubit_hamiltonian_dict)
print(f"Final Hamiltonian: {combined_hamiltonian_final}")
# Take the combined_hamiltonian_final, put this inside hamiltonian.py and proceed.



# print(*[value.real for value in combined_hamiltonian_dict.values()])
# Add distance infront of this list called real_parts
real_parts = [f"{value.real:.10f}" for value in combined_hamiltonian_dict.values()]
# print(f"Real parts: {str(real_parts)}")
# fout.write(" ".join(str(real) for real in real_parts))
# fout.write("\n")
# fout.write("FCI gs energy %.6f \n" % combined_hamiltonian_final)
# Calculate ground state of Hamiltonian
h_sparse = qubit_operator_sparse(qubit_hamiltonian)
s2_sparse = qubit_operator_sparse(s2_qubit_op)
gs_ene, gs_wf = get_ground_state(h_sparse)
s2val = expectation(s2_sparse, gs_wf).real


# # For Hydrogen molecule
# if not bs:
#     if mapping_method == 'JWT':
#         hf_config = [1, 1, 0, 0]
#     elif mapping_method == 'BKT':
#         hf_config = [1, 0, 0, 0]
#     elif mapping_method == 'SCBKT':
#         hf_config = [1, 1]
#     #
#     hf_wf = zeros((2**nq), dtype=np.complex64)
#     hf_pointer = 0
#     for iq in range(nq):
#         if hf_config[iq] == 1:
#             hf_pointer += 2**(nq-iq-1)
#     hf_wf[hf_pointer] = 1.0+0.0j
#     #
#     hf_energy = expectation(h_sparse, hf_wf).real
#     hf_s2val = expectation(s2_sparse, hf_wf).real
#     print('\n E(HF)  = {:.10f}'.format(hf_energy), 'Hartree, <S^2> = {:.4f}'.format(hf_s2val))
# else:
#     if mapping_method == 'JWT':
#         bs_config = [1, 0, 0, 1]
#     elif mapping_method == 'BKT':
#         bs_config = [1, 1, 0, 0]
#     elif mapping_method == 'SCBKT':
#         bs_config = [1, 0]
#     #
#     bs_wf = zeros((2**nq), dtype=np.complex64)
#     bs_pointer = 0
#     for iq in range(nq):
#         if bs_config[iq] == 1:
#             bs_pointer += 2**(nq-iq-1)
#     bs_wf[bs_pointer] = 1.0+0.0j
#     print(f"Broken Symmetry Initial Wave-function: {bs_wf}")
#     #
#     bs_energy = expectation(h_sparse, bs_wf).real
#     bs_s2val = expectation(s2_sparse, bs_wf).real
#     print('\n E(BS)  = {:.10f}'.format(bs_energy), 'Hartree, <S^2> = {:.4f}'.format(bs_s2val))
# #
# print(' E(FCI) = {:.10f}'.format(gs_ene), 'Hartree, <S^2> = {:.4f}'.format(s2val))
# print('\n Full-CI wave function')
# print_wf(gs_wf, nq)

# fout.close()
