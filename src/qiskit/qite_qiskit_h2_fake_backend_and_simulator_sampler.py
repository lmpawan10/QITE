import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from scipy import linalg as SciLA

# Test imports
try:
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2
    from qiskit.primitives import BackendSampler
    FAKE_BACKEND_AVAILABLE = True
except ImportError:
    FAKE_BACKEND_AVAILABLE = False
    print("Warning: Fake backend not available. Using simulator only.")

# Index mapping for 2-qubit Pauli products
def get_pauli_index(pauli_string):
    """Convert Pauli string to index (0-15 for 2 qubits)"""
    pauli_map = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}
    return pauli_map[pauli_string[0]] * 4 + pauli_map[pauli_string[1]]

def get_pauli_strings_2qubit():
    """Generate all 16 Pauli strings for 2 qubits"""
    pauli_chars = ['I', 'X', 'Y', 'Z']
    pauli_strings = []
    for p0 in pauli_chars:
        for p1 in pauli_chars:
            pauli_strings.append(p0 + p1)
    return pauli_strings

def get_pauli_product_rules():
    """Get multiplication rules for single-qubit Paulis"""
    rules = {}
    for p in ['I', 'X', 'Y', 'Z']:
        rules[('I', p)] = (p, 1)
        rules[(p, 'I')] = (p, 1)
    
    rules[('X', 'X')] = ('I', 1)
    rules[('X', 'Y')] = ('Z', 1j)
    rules[('X', 'Z')] = ('Y', -1j)
    rules[('Y', 'X')] = ('Z', -1j)
    rules[('Y', 'Y')] = ('I', 1)
    rules[('Y', 'Z')] = ('X', 1j)
    rules[('Z', 'X')] = ('Y', 1j)
    rules[('Z', 'Y')] = ('X', -1j)
    rules[('Z', 'Z')] = ('I', 1)
    
    return rules

def multiply_pauli_strings(p1, p2):
    """Multiply two 2-qubit Pauli strings"""
    rules = get_pauli_product_rules()
    result0, coeff0 = rules[(p1[0], p2[0])]
    result1, coeff1 = rules[(p1[1], p2[1])]
    return result0 + result1, coeff0 * coeff1

def ansatz(qc, qbits):
    """Define the initial ansatz circuit"""
    qc.x(qbits[1])  # Start from |01⟩ state

def propagate(qc, alist, qbits):
    """Apply the imaginary time evolution operator based on alist"""
    if len(alist) == 0:
        return
    
    pauli_strings = get_pauli_strings_2qubit()
    
    for t in range(len(alist)):
        for i, pauli_string in enumerate(pauli_strings):
            angle = np.real(alist[t][i])
            if abs(angle) > 1e-10 and pauli_string != 'II':
                apply_pauli_rotation(qc, pauli_string, angle, qbits)

def apply_pauli_rotation(qc, pauli_string, angle, qbits):
    """Apply exp(-i * angle * Pauli_string / 2) using gates"""
    # Convert to appropriate basis
    for i, p in enumerate(pauli_string):
        if p == 'X':
            qc.h(qbits[i])
        elif p == 'Y':
            qc.rx(np.pi/2, qbits[i])
    
    # Apply the rotation
    active_qubits = [i for i, p in enumerate(pauli_string) if p != 'I']
    
    if len(active_qubits) == 1:
        qc.rz(angle, qbits[active_qubits[0]])
    elif len(active_qubits) == 2:
        qc.cx(qbits[active_qubits[0]], qbits[active_qubits[1]])
        qc.rz(angle, qbits[active_qubits[1]])
        qc.cx(qbits[active_qubits[0]], qbits[active_qubits[1]])
    
    # Convert back from basis
    for i, p in enumerate(pauli_string):
        if p == 'X':
            qc.h(qbits[i])
        elif p == 'Y':
            qc.rx(-np.pi/2, qbits[i])

def measure_pauli_expectation(alist, shots, backend, qbits, pauli_string, sampler=None):
    """Measure expectation value of a Pauli operator"""
    print(f"Inside measure_pauli_expectation")
    if pauli_string == 'II':
        return 1.0
    
    qr = QuantumRegister(len(qbits))
    cr = ClassicalRegister(len(qbits))
    qc = QuantumCircuit(qr, cr)
    
    ansatz(qc, qr)
    propagate(qc, alist, qr)
    
    # Add basis rotations
    for i, pauli_char in enumerate(pauli_string):
        if pauli_char == 'X':
            qc.h(qr[i])
        elif pauli_char == 'Y':
            qc.sdg(qr[i])
            qc.h(qr[i])
    
    # Add measurements
    qc.measure_all()
    
    # Transpile the circuit
    qc_transpiled = transpile(qc, backend, optimization_level=2)
    
    # Use traditional execution (works for both simulator and fake backend)
    job = backend.run(qc_transpiled, shots=shots)
    result = job.result()
    counts = result.get_counts()
    
    # Calculate expectation value
    expectation = 0
    total = sum(counts.values())
    
    for bitstring, count in counts.items():
        # Clean bitstring (remove spaces)
        bitstring = bitstring.replace(' ', '')
        
        # Calculate parity
        parity = 1
        for i, pauli_char in enumerate(pauli_string):
            if pauli_char != 'I':
                # Qiskit bit ordering: rightmost bit is qubit 0
                # So we need to reverse the indexing
                bit_index = len(qbits) - 1 - i
                if bitstring[bit_index] == '1':
                    parity *= -1
        
        expectation += parity * count / total
    
    return expectation

def measure_energy(alist, shots, backend, qbits, hm_list):
    """Measure the energy at the end of each time step"""
    energy = 0
    pauli_strings = get_pauli_strings_2qubit()
    
    for i in range(len(hm_list)):
        for hm in hm_list[i]:
            pauli_indices = hm[0]
            coefficients = hm[1]
            
            for j in range(len(pauli_indices)):
                pauli_idx = pauli_indices[j]
                coeff = coefficients[j]
                pauli_string = pauli_strings[pauli_idx]
                expectation = measure_pauli_expectation(alist, shots, backend, qbits, pauli_string)
                energy += coeff * expectation
    
    return energy

def get_expectation(alist, shots, backend, qbits):
    """Obtain the expectation values of all 16 Pauli operators"""
    pauli_strings = get_pauli_strings_2qubit()
    sigma_expectation = np.zeros(16, dtype=complex)
    
    for i, pauli_string in enumerate(pauli_strings):
        sigma_expectation[i] = measure_pauli_expectation(alist, shots, backend, qbits, pauli_string)
    
    return sigma_expectation

def get_S_matrix_2qubit(sigma_expectation):
    """Construct S matrix for 2-qubit system"""
    pauli_strings = get_pauli_strings_2qubit()
    S = np.zeros((16, 16), dtype=complex)
    
    for i in range(16):
        for j in range(16):
            pauli_i = pauli_strings[i]
            pauli_j = pauli_strings[j]
            result_pauli, coeff = multiply_pauli_strings(pauli_i, pauli_j)
            result_idx = get_pauli_index(result_pauli)
            S[i, j] = coeff * sigma_expectation[result_idx]
    
    return S

def update_alist(sigma_expectation, alist, db, delta, hm):
    """Update the a-list using McLachlan's variational principle for 2 qubits"""
    S = get_S_matrix_2qubit(sigma_expectation)
    b = np.zeros(16, dtype=complex)
    
    # Calculate normalization factor c
    c = 1
    pauli_indices = hm[0][0]
    coefficients = hm[0][1]
    for i in range(len(pauli_indices)):
        pauli_idx = pauli_indices[i]
        coeff = coefficients[i]
        c -= 2 * db * coeff * sigma_expectation[pauli_idx]
    c = np.sqrt(abs(c))
    
    # Calculate b vector components
    pauli_strings = get_pauli_strings_2qubit()
    for i in range(16):
        b[i] += (sigma_expectation[i] / c - sigma_expectation[i]) / db
        
        for j in range(len(pauli_indices)):
            h_idx = pauli_indices[j]
            h_coeff = coefficients[j]
            h_pauli = pauli_strings[h_idx]
            
            pauli_i = pauli_strings[i]
            result_pauli, prod_coeff = multiply_pauli_strings(pauli_i, h_pauli)
            result_idx = get_pauli_index(result_pauli)
            
            b[i] -= h_coeff * prod_coeff * sigma_expectation[result_idx] / c
        
        b[i] = 1j * b[i] - 1j * np.conj(b[i])
    
    # Add regularizer
    dalpha = np.eye(16) * delta
    
    # Solve linear equation
    x = SciLA.lstsq(S + np.transpose(S) + dalpha, -b)[0]
    
    # Append new angles to alist
    alist.append([])
    for i in range(16):
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
    
    E[0] = measure_energy(alist, shots, backend, qbits, hm_list)
    print(f"Initial energy: {E[0].real:.6f}")
    
    for i in range(1, N+1):
        alist = qite_step(alist, shots, backend, qbits, db, delta, hm_list)
        E[i] = measure_energy(alist, shots, backend, qbits, hm_list)
        
        if i % 5 == 0:
            print(f"Step {i}: Energy = {E[i].real:.6f}")
    
    return E

if __name__ == '__main__':
    # Parameters
    N = 25
    shots = 5000
    db = 0.1
    delta = 0.1
    
    # Test with fake backend
    use_fake = True and FAKE_BACKEND_AVAILABLE
    
    if use_fake:
        print("Using Fake Backend")
        backend = FakeManilaV2()
        print(f"Backend: {backend.name}")
        print(f"Qubits: {backend.num_qubits}")
    else:
        print("Using AerSimulator")
        backend = AerSimulator()
    
    qbits = [0, 1]
    
    # Define Hamiltonian
    coeffs = [-0.3104, 0.1026, 0.0632, 0.3406, 0.1450, 0.1450]
    paulis = ["II", "ZI", "IZ", "ZZ", "XX", "YY"]
    
    # Convert to hm_list format
    hm_list = []
    for pauli, coeff in zip(paulis, coeffs):
        hm_list.append([])
        pauli_idx = get_pauli_index(pauli)
        hm_list[-1].append([[pauli_idx], [coeff]])
    
    print(f"\n2-Qubit Hamiltonian:")
    for pauli, coeff in zip(paulis, coeffs):
        print(f"  {coeff:+.4f} * {pauli}")
    
    # Calculate exact ground state
    pauli_matrices = {
        'I': np.array([[1, 0], [0, 1]], dtype=complex),
        'X': np.array([[0, 1], [1, 0]], dtype=complex),
        'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'Z': np.array([[1, 0], [0, -1]], dtype=complex)
    }
    
    H = np.zeros((4, 4), dtype=complex)
    for pauli, coeff in zip(paulis, coeffs):
        H_term = np.kron(pauli_matrices[pauli[0]], pauli_matrices[pauli[1]])
        H += coeff * H_term
    
    eigenvalues, _ = np.linalg.eigh(H)
    print(f"\nExact ground state energy: {eigenvalues[0]:.6f}")
    
    # Test initial state
    print(f"\nTesting initial state |01⟩:")
    test_energy = 0
    for pauli, coeff in zip(paulis, coeffs):
        exp_val = measure_pauli_expectation([], shots, backend, qbits, pauli)
        print(f"  ⟨{pauli}⟩ = {exp_val:+.4f}")
        test_energy += coeff * exp_val
    print(f"Initial energy: {test_energy:.6f}")
    
    # Run QITE
    print(f"\nRunning QITE with {N} steps...\n")
    E = qite(backend, qbits, shots, db, delta, N, hm_list)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    time_steps = np.arange(N+1) * db
    plt.plot(time_steps, E.real, 'b-', linewidth=2, label='QITE Energy')
    plt.axhline(y=eigenvalues[0], color='r', linestyle='--', label=f'Ground State ({eigenvalues[0]:.4f})')
    plt.xlabel('Imaginary Time β', fontsize=12)
    plt.ylabel('Energy', fontsize=12)
    plt.title(f'2-Qubit QITE ({backend.name})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nFinal energy: {E[-1].real:.6f}")
    print(f"Exact ground state: {eigenvalues[0]:.6f}")
    print(f"Error: {abs(E[-1].real - eigenvalues[0]):.6f}")