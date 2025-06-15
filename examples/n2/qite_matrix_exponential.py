import openfermion
import numpy as np
import scipy as sp
import datetime
import time

from math import sqrt
from numpy import zeros, dot, conjugate
from scipy.sparse.linalg import expm_multiply
from openfermion.config import *
from openfermion.ops import *
from openfermion.transforms import *
from openfermion.utils import *
from openfermion.linalg import *

from ...src.inputs import get_n2_bs2_integral_files_path, get_n2_bs3_integral_files_path, get_n2_integral_files_path

# System specification
integral_file = get_n2_integral_files_path()

nso = 12  # number of spin orbitals
ne  = 6  # number of electron

# scbkt = False   # If False, then use Jordan-Wigner with interleaved arrangement
# ref_config = '111111000000' # (HF, JW) bitstring corresponding to the reference state 
# ref_config = '111010010100' # (BS2, JW) bitstring corresponding to the reference state 
# ref_config = '101010010101' # (BS3, JW) bitstring corresponding to the reference state 

scbkt = True
ref_config = '101010000000' # (HF, BKT) bitstring corresponding to the reference state 
# ref_config = '101111010101' # (BS2, BKT) bitstring corresponding to the reference state 
# ref_config = '111011000100' # (BS3, BKT) bitstring corresponding to the reference state 

use_penalty = False

# QITE condition
dt = 0.1  # evolution time length for single step
nsteps = 100 # maximum time steps
coef_s2 = 1.0  # H' = H + coef * S^2 operator

#---------- FUNCTION GET_INTEGRALS ----------#
def get_hamiltonian(integral_file):
    hamiltonian = FermionOperator()
    for line in open(integral_file):
        items = line.split()
        if len(items) == 1:
            intval = float(items[0])
            hamiltonian += FermionOperator((), intval)
        elif len(items) == 3:
            # One electron terms
            iocc = int(items[0])
            avir = int(items[1])
            intval = float(items[2])
            hamiltonian += FermionOperator(((avir, 1), (iocc, 0)), intval)
        elif len(items) == 5:
            # Two electron terms
            iocc = int(items[0])
            jocc = int(items[1])
            bvir = int(items[2])
            avir = int(items[3])
            intval = float(items[4])
            hamiltonian += FermionOperator(((avir, 1), (bvir, 1), (jocc, 0), (iocc, 0)), intval)
    return hamiltonian

#---------- FUNCTION S_SQUARED_FERMION_DM ----------#
def get_s2_operator(nso):
    # generate S^2 Fermionic operator in DM.
    """
    Notes:
    S(i,j)^2 = S_z(i)*S_z(j) + (S_+(i) * S_-(j) + S_-(i) * S_+(j))/2
    """
    nmo = int(nso / 2)
    s2_operator = FermionOperator()
     
    for iorb in range(nmo):
        ia = 2 * iorb
        ib  = 2 * iorb + 1
        for jorb in range(nmo):
            ja = 2 * jorb
            jb  = 2 * jorb + 1
            # S_z(i) * S_z(j) terms
            s2_operator +=  0.25 * FermionOperator(((ia, 1), (ia, 0), (ja, 1), (ja, 0)))
            s2_operator += -0.25 * FermionOperator(((ia, 1), (ia, 0), (jb, 1), (jb, 0)))
            s2_operator += -0.25 * FermionOperator(((ib, 1), (ib, 0), (ja, 1), (ja, 0)))
            s2_operator +=  0.25 * FermionOperator(((ib, 1), (ib, 0), (jb, 1), (jb, 0)))
            # (S_+(i) * S_-(j) + S_-(i) * S_+(j))/2 terms
            s2_operator +=  0.50 * FermionOperator(((ia, 1), (ib, 0), (jb, 1), (ja, 0)))
            s2_operator +=  0.50 * FermionOperator(((ib, 1), (ia, 0), (ja, 1), (jb, 0)))    
    return s2_operator

##########
hamiltonian = get_hamiltonian(integral_file)
s2_operator = get_s2_operator(nso)

if scbkt == True:
    # nq = nso - 2
    nq = nso
    # hq = symmetry_conserving_bravyi_kitaev(hamiltonian, nso, ne)
    # s2_qubop = symmetry_conserving_bravyi_kitaev(s2_operator, nso, ne)
    hq = bravyi_kitaev(hamiltonian, nq)
    s2_qubop = bravyi_kitaev(s2_operator, nq)
else:
    nq = nso
    hq = jordan_wigner(hamiltonian)
    s2_qubop = jordan_wigner(s2_operator)
    
hq_sparse = qubit_operator_sparse(hq)
s2_sparse = qubit_operator_sparse(s2_qubop)

if use_penalty:
    h_for_qite = hq_sparse + coef_s2 * s2_sparse
else:
    h_for_qite = hq_sparse

##########
ref_state = zeros((2**nq))
ref_pointer = 0
for iq in range(nq):
    if ref_config[iq:iq+1] == '1':
        ref_pointer += 2**(nq-iq-1)
ref_state[ref_pointer] = 1.0

# CAS-CI calculation for the reference
gs_ene, gs_wf = get_ground_state(h_for_qite)
print(f"gs_ene: {gs_ene}")

# from scipy.sparse.linalg import eigsh
# gs_ene, gs_wf = eigsh(hq_sparse,k=1,which="SA")

# initial state
print(' Time  E(QITE)/Hartree   |<QITE|CAS>|^2   Delta-E/kcal mol-1   <S^2>')
ref_energy = expectation(h_for_qite, ref_state).real
ref_s2 = expectation(s2_sparse, ref_state).real
ref_overlap = dot(ref_state, conjugate(gs_wf))
ref_fidelity = dot(ref_overlap, conjugate(ref_overlap)).real
print('  0.0    {:.10f}'.format(ref_energy),'      {:.6f}'.format(ref_fidelity),
          '        {:.3f}'.format(627.51*(ref_energy-gs_ene)),'            {:.4f}'.format(ref_s2))
# Imaginary time evolution start
curr_state = ref_state
for istep in range(nsteps):
    qite_state_unnormalized = expm_multiply(-1*h_for_qite, curr_state)
    norm_factor = sqrt(1/dot(qite_state_unnormalized, conjugate(qite_state_unnormalized)).real)
    qite_state_normalized = norm_factor * qite_state_unnormalized
    # Calc energy, fidelity, and <S^2>
    energy = expectation(h_for_qite, qite_state_normalized).real
    overlap = dot(gs_wf, conjugate(qite_state_normalized))
    fidelity = dot(overlap, conjugate(overlap)).real
    s2val = expectation(s2_sparse, qite_state_normalized).real
    print('  {:.1f}'.format(dt*(istep+1)),'   {:.10f}'.format(energy),'      {:.6f}'.format(fidelity),
          '        {:.3f}'.format(627.51*(energy-gs_ene)),'            {:.4f}'.format(s2val))
    curr_state = qite_state_normalized
