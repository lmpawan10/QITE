import numpy as np
import matplotlib.pyplot as plt
import cirq
from scipy import linalg as SciLA

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

def ansatz(circuit, qubits):
    """Define the initial ansatz circuit"""
    # Start from |01⟩ state
    circuit.append(cirq.X(qubits[1]))

def propagate(circuit, alist, qubits):
    """Apply the imaginary time evolution operator based on alist"""
    if len(alist) == 0:
        return
    
    pauli_strings = get_pauli_strings_2qubit()
    
    for t in range(len(alist)):
        for i, pauli_string in enumerate(pauli_strings):
            angle = np.real(alist[t][i])
            if abs(angle) > 1e-10 and pauli_string != 'II':
                apply_pauli_rotation(circuit, pauli_string, angle, qubits)

def apply_pauli_rotation(circuit, pauli_string, angle, qubits):
    """Apply exp(-i * angle * Pauli_string / 2) using gates"""
    # Convert to appropriate basis
    for i, p in enumerate(pauli_string):
        if p == 'X':
            circuit.append(cirq.H(qubits[i]))
        elif p == 'Y':
            circuit.append(cirq.rx(np.pi/2)(qubits[i]))
    
    # Apply the rotation
    active_qubits = [i for i, p in enumerate(pauli_string) if p != 'I']
    
    if len(active_qubits) == 1:
        circuit.append(cirq.rz(angle)(qubits[active_qubits[0]]))
    elif len(active_qubits) == 2:
        circuit.append(cirq.CNOT(qubits[active_qubits[0]], qubits[active_qubits[1]]))
        circuit.append(cirq.rz(angle)(qubits[active_qubits[1]]))
        circuit.append(cirq.CNOT(qubits[active_qubits[0]], qubits[active_qubits[1]]))
    
    # Convert back from basis
    for i, p in enumerate(pauli_string):
        if p == 'X':
            circuit.append(cirq.H(qubits[i]))
        elif p == 'Y':
            circuit.append(cirq.rx(-np.pi/2)(qubits[i]))

def get_pauli_operator(pauli_string, qubits):
    """Create a Cirq PauliString from a string representation"""
    pauli_map = {
        'I': cirq.I,
        'X': cirq.X,
        'Y': cirq.Y,
        'Z': cirq.Z
    }
    
    terms = []
    for i, p in enumerate(pauli_string):
        if p != 'I':
            terms.append(pauli_map[p](qubits[i]))
    
    if len(terms) == 0:
        return cirq.PauliString()  # Identity
    elif len(terms) == 1:
        return cirq.PauliString(terms[0])
    else:
        return cirq.PauliString(terms[0], terms[1])

def measure_pauli_expectation(alist, shots, simulator, qubits, pauli_string):
    """Measure expectation value of a Pauli operator"""
    if pauli_string == 'II':
        return 1.0
    
    # Create circuit
    circuit = cirq.Circuit()
    ansatz(circuit, qubits)
    propagate(circuit, alist, qubits)
    
    # Create the observable
    observable = get_pauli_operator(pauli_string, qubits)
    
    # Add measurement basis rotations
    for i, pauli_char in enumerate(pauli_string):
        if pauli_char == 'X':
            circuit.append(cirq.H(qubits[i]))
        elif pauli_char == 'Y':
            circuit.append(cirq.S(qubits[i])**-1)  # S^dagger
            circuit.append(cirq.H(qubits[i]))
    
    # Add measurements
    circuit.append(cirq.measure(*qubits, key='result'))
    
    # Run the circuit
    result = simulator.run(circuit, repetitions=shots)
    measurements = result.measurements['result']
    
    # Calculate expectation value
    expectation = 0
    for measurement in measurements:
        parity = 1
        for i, pauli_char in enumerate(pauli_string):
            if pauli_char != 'I':
                if measurement[i] == 1:
                    parity *= -1
        expectation += parity
    
    return expectation / shots

def measure_energy(alist, shots, simulator, qubits, hm_list):
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
                expectation = measure_pauli_expectation(alist, shots, simulator, qubits, pauli_string)
                energy += coeff * expectation
    
    return energy

def get_expectation(alist, shots, simulator, qubits):
    """Obtain the expectation values of all 16 Pauli operators"""
    pauli_strings = get_pauli_strings_2qubit()
    sigma_expectation = np.zeros(16, dtype=complex)
    
    for i, pauli_string in enumerate(pauli_strings):
        sigma_expectation[i] = measure_pauli_expectation(alist, shots, simulator, qubits, pauli_string)
    
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

def qite_step(alist, shots, simulator, qubits, db, delta, hm_list):
    """Perform one QITE step"""
    for j in range(len(hm_list)):
        sigma_expectation = get_expectation(alist, shots, simulator, qubits)
        norm = update_alist(sigma_expectation, alist, db, delta, hm_list[j])
    return alist

def qite(simulator, qubits, shots, db, delta, N, hm_list):
    """Main QITE algorithm"""
    E = np.zeros(N+1, dtype=complex)
    alist = []
    
    E[0] = measure_energy(alist, shots, simulator, qubits, hm_list)
    print(f"Initial energy: {E[0].real:.6f}")
    
    for i in range(1, N+1):
        alist = qite_step(alist, shots, simulator, qubits, db, delta, hm_list)
        E[i] = measure_energy(alist, shots, simulator, qubits, hm_list)
        
        if i % 5 == 0:
            print(f"Step {i}: Energy = {E[i].real:.6f}")
            print(f"A_list length: {len(alist)}")
    
    return E, alist

if __name__ == '__main__':
    # Parameters
    N = 25
    shots = 5000
    db = 0.1
    delta = 0.1
    
    # Create simulator
    simulator = cirq.Simulator()
    
    # Create qubits
    qubits = cirq.LineQubit.range(2)
    print(f"Using Cirq Simulator with qubits: {qubits}")
    
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
    
    # Calculate exact ground state using Cirq
    pauli_terms = []
    for pauli, coeff in zip(paulis, coeffs):
        if pauli != "II":
            op = get_pauli_operator(pauli, qubits)
            pauli_terms.append(coeff * op)
        else:
            pauli_terms.append(coeff * cirq.PauliString())  # Identity term
    
    # Create the Hamiltonian as a sum of Pauli strings
    H_matrix = np.zeros((4, 4), dtype=complex)
    pauli_matrices = {
        'I': np.array([[1, 0], [0, 1]], dtype=complex),
        'X': np.array([[0, 1], [1, 0]], dtype=complex),
        'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'Z': np.array([[1, 0], [0, -1]], dtype=complex)
    }
    
    for pauli, coeff in zip(paulis, coeffs):
        H_term = np.kron(pauli_matrices[pauli[0]], pauli_matrices[pauli[1]])
        H_matrix += coeff * H_term
    
    eigenvalues, _ = np.linalg.eigh(H_matrix)
    print(f"\nExact ground state energy: {eigenvalues[0]:.6f}")
    
    # Test initial state
    print(f"\nTesting initial state |01⟩:")
    test_energy = 0
    for pauli, coeff in zip(paulis, coeffs):
        exp_val = measure_pauli_expectation([], shots, simulator, qubits, pauli)
        print(f"  ⟨{pauli}⟩ = {exp_val:+.4f}")
        test_energy += coeff * exp_val
    print(f"Initial energy: {test_energy:.6f}")
    
    # Run QITE
    print(f"\nRunning QITE with {N} steps...\n")
    E, alist = qite(simulator, qubits, shots, db, delta, N, hm_list)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    time_steps = np.arange(N+1) * db
    plt.plot(time_steps, E.real, 'b-', linewidth=2, label='QITE Energy')
    plt.axhline(y=eigenvalues[0], color='r', linestyle='--', label=f'Ground State ({eigenvalues[0]:.4f})')
    plt.xlabel('Imaginary Time β', fontsize=12)
    plt.ylabel('Energy', fontsize=12)
    plt.title('2-Qubit QITE (Cirq)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nFinal energy: {E[-1].real:.6f}")
    print(f"Exact ground state: {eigenvalues[0]:.6f}")
    print(f"Error: {abs(E[-1].real - eigenvalues[0]):.6f}")
    
    # Show example circuit
    print("\n" + "="*50)
    print("Example: Final QITE circuit (first few gates):")
    print("="*50)
    example_circuit = cirq.Circuit()
    ansatz(example_circuit, qubits)
    
    # Show just first step of propagation if alist is not empty
    if len(alist) > 0:
        pauli_strings = get_pauli_strings_2qubit()
        gates_shown = 0
        for i, pauli_string in enumerate(pauli_strings):
            if gates_shown >= 5:  # Show only first 5 non-trivial gates
                break
            angle = np.real(alist[0][i])
            if abs(angle) > 1e-10 and pauli_string != 'II':
                apply_pauli_rotation(example_circuit, pauli_string, angle, qubits)
                gates_shown += 1
    
    print(example_circuit)