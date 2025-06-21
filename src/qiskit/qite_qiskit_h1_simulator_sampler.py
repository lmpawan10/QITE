import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

# Index mapping for Pauli products (from index.py equivalent)
def get_idx_coeff():
    """Define the index and coefficient mappings for Pauli products"""
    idx = np.zeros((4, 4), dtype=int)
    coeff = np.zeros((4, 4), dtype=complex)
    
    # I ⊗ I = I, I ⊗ X = X, I ⊗ Y = Y, I ⊗ Z = Z
    idx[0, :] = [0, 1, 2, 3]
    coeff[0, :] = [1, 1, 1, 1]
    
    # X ⊗ I = X, X ⊗ X = I, X ⊗ Y = iZ, X ⊗ Z = -iY
    idx[1, :] = [1, 0, 3, 2]
    coeff[1, :] = [1, 1, 1j, -1j]
    
    # Y ⊗ I = Y, Y ⊗ X = -iZ, Y ⊗ Y = I, Y ⊗ Z = iX
    idx[2, :] = [2, 3, 0, 1]
    coeff[2, :] = [1, -1j, 1, 1j]
    
    # Z ⊗ I = Z, Z ⊗ X = iY, Z ⊗ Y = -iX, Z ⊗ Z = I
    idx[3, :] = [3, 2, 1, 0]
    coeff[3, :] = [1, 1j, -1j, 1]
    
    return idx, coeff

def ansatz(qc, qbits):
    """Define the initial ansatz circuit"""
    # Start from |0⟩ state (default)
    pass

def measure(qc, pauli_index, qbit):
    """Add measurement in different Pauli bases to circuit
    0: I (identity)
    1: X basis
    2: Y basis  
    3: Z basis
    """
    if pauli_index == 0:  # Identity
        pass  # No measurement needed
    elif pauli_index == 1:  # X basis
        qc.h(qbit)
        qc.measure(qbit, 0)
    elif pauli_index == 2:  # Y basis
        qc.rx(np.pi/2, qbit)  # RX(π/2) to rotate Y to Z
        qc.measure(qbit, 0)
    elif pauli_index == 3:  # Z basis
        qc.measure(qbit, 0)

def propagate(qc, alist, qbits):
    """Apply the imaginary time evolution operator based on alist"""
    if len(alist) == 0:
        return
    
    for t in range(len(alist)):
        # Each element in alist contains 4 angles (for I, X, Y, Z)
        # We only apply rotations for X, Y, Z (indices 1, 2, 3)
        for gate in range(1, 4):
            angle = np.real(alist[t][gate])
            if gate == 1:
                qc.rx(angle, qbits[0])
            elif gate == 2:
                qc.ry(angle, qbits[0])
            elif gate == 3:
                qc.rz(angle, qbits[0])

def measure_pauli_expectation(alist, shots, backend, qbits, pauli_index):
    """Measure expectation value of a Pauli operator"""
    if pauli_index == 0:  # Identity
        return 1.0
    
    qr = QuantumRegister(len(qbits))
    cr = ClassicalRegister(1)
    qc = QuantumCircuit(qr, cr)
    
    ansatz(qc, qr)
    propagate(qc, alist, qr)
    measure(qc, pauli_index, qr[0])
    
    # Execute circuit
    job = backend.run(qc, shots=shots)
    counts = job.result().get_counts()
    
    # Calculate probabilities
    total_shots = sum(counts.values())
    p0 = counts.get('0', 0) / total_shots
    p1 = counts.get('1', 0) / total_shots
    
    # Return expectation value: P(0) - P(1)
    return p0 - p1

def measure_energy(alist, shots, backend, qbits, hm_list):
    """Measure the energy at the end of each time step"""
    energy = 0
    for i in range(len(hm_list)):
        for hm in hm_list[i]:
            for j in range(len(hm[0])):
                expectation = measure_pauli_expectation(alist, shots, backend, 
                                                       qbits, hm[0][j])
                energy += hm[1][j] * expectation
    return energy

def get_expectation(alist, shots, backend, qbits):
    """Obtain the expectation values of all Pauli operators"""
    sigma_expectation = np.zeros(4, dtype=complex)
    for j in range(4):
        sigma_expectation[j] = measure_pauli_expectation(alist, shots, backend, qbits, j)
    return sigma_expectation

def update_alist(sigma_expectation, alist, db, delta, hm):
    """Update the a-list using McLachlan's variational principle"""
    idx, coeff = get_idx_coeff()
    
    # Step 1: Construct S matrix
    S = np.zeros((4, 4), dtype=complex)
    for i in range(4):
        for j in range(4):
            S[i, j] = sigma_expectation[idx[i, j]] * coeff[i, j]
    
    # Step 2: Construct b vector
    b = np.zeros(4, dtype=complex)
    
    # Calculate normalization factor c
    c = 1
    for i in range(len(hm[0][0])):
        c -= 2 * db * hm[0][1][i] * sigma_expectation[hm[0][0][i]]
    c = np.sqrt(c)
    
    # Calculate b vector components
    for i in range(4):
        b[i] += (sigma_expectation[i] / c - sigma_expectation[i]) / db
        for j in range(len(hm[0][0])):
            b[i] -= hm[0][1][j] * coeff[i, hm[0][0][j]] * sigma_expectation[idx[i, hm[0][0][j]]] / c
        b[i] = 1j * b[i] - 1j * np.conj(b[i])
    
    # Step 3: Add regularizer
    dalpha = np.eye(4) * delta
    
    # Step 4: Solve linear equation
    # The solution is multiplied by -2 because unitary rotation gates are exp(-i*theta/2)
    x = np.linalg.lstsq(S + np.transpose(S) + dalpha, -b, rcond=None)[0]
    
    # Append new angles to alist
    alist.append([])
    for i in range(len(x)):
        alist[-1].append(-x[i] * 2 * db)
    
    return c

def qite_step(alist, shots, backend, qbits, db, delta, hm_list):
    """Perform one QITE step"""
    for j in range(len(hm_list)):
        sigma_expectation = get_expectation(alist, shots, backend, qbits)
        norm = update_alist(sigma_expectation, alist, db, delta, hm_list[j])
    return alist

def qite(backend, qbits, shots, db, delta, N, hm_list):
    """Main QITE algorithm"""
    E = np.zeros(N+1, dtype=complex)
    alist = []
    
    # Initial energy
    E[0] = measure_energy(alist, shots, backend, qbits, hm_list)
    print(f"Initial energy: {E[0].real:.6f}")
    
    # QITE main loop
    for i in range(1, N+1):
        alist = qite_step(alist, shots, backend, qbits, db, delta, hm_list)
        E[i] = measure_energy(alist, shots, backend, qbits, hm_list)
        
        if i % 5 == 0:
            print(f"Step {i}: Energy = {E[i].real:.6f}")
            print(f"A_list length: {len(alist)}")
    
    return E, alist

if __name__ == '__main__':
    # Input parameters for QITE
    N = 25
    shots = 1000
    db = 0.1
    delta = 0.1
    
    # Initialize Qiskit backend
    backend = AerSimulator()
    qbits = [0]
    
    # Define Hamiltonian terms
    # H = (1/√2) * X + (1/√2) * Z
    hm_list = []
    hm_list.append([])
    hm_list[0].append([[1], [1/np.sqrt(2)]])  # X term
    hm_list.append([])
    hm_list[1].append([[3], [1/np.sqrt(2)]])  # Z term
    
    print(f"Hamiltonian: H = (1/√2)X + (1/√2)Z")
    print(f"Theoretical ground state energy: -1.0")
    print(f"Running QITE with {N} steps...\n")
    
    # Run QITE
    E, alist = qite(backend, qbits, shots, db, delta, N, hm_list)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    time_steps = np.arange(N+1) * db  # Note: using db for time steps as in original
    plt.plot(time_steps, E.real, 'b-', linewidth=2, label='QITE Energy')
    plt.axhline(y=-1.0, color='r', linestyle='--', label='Ground State Energy')
    plt.xlabel('Imaginary Time β', fontsize=12)
    plt.ylabel('Energy', fontsize=12)
    plt.title('Quantum Imaginary Time Evolution', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.ylim(-1.2, 1.0)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nFinal energy: {E[-1].real:.6f}")
    print(f"Error from ground state: {abs(E[-1].real - (-1.0)):.6f}")
    
    # Print final state analysis
    final_exp = get_expectation(alist, shots, backend, qbits)
    print(f"\nFinal state expectation values:")
    print(f"<I> = {final_exp[0].real:.6f}")
    print(f"<X> = {final_exp[1].real:.6f}")
    print(f"<Y> = {final_exp[2].real:.6f}")
    print(f"<Z> = {final_exp[3].real:.6f}")