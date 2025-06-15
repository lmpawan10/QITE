import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply
from scipy import linalg as SciLA
import matplotlib.pyplot as plt

from inputs import get_backend, get_channel, get_ibm_token, get_instance

import qiskit
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector, Pauli, Operator
from qiskit.primitives import StatevectorEstimator, Sampler, Estimator
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.circuit.library import PauliEvolutionGate
from qiskit_aer import AerSimulator
from qiskit.synthesis.evolution import LieTrotter

from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2 as Estimator, QiskitRuntimeService, SamplerV2 as Sampler

QiskitRuntimeService.save_account(channel=get_channel(), token=get_ibm_token(), overwrite=True, set_as_default=True)
service = QiskitRuntimeService(channel=get_channel(), instance=get_instance())
real_backend = service.backend(get_backend())

xv = []
db = 0.1
bmax = 20
nsteps = bmax/db
nspin = 2

distance = 2.7
basis = "sto-3g"
spin = 0
charge = 0

psi = np.zeros(2**nspin, dtype=complex)


# HF
# This would be 11 or 00 otherwise if the hamiltonian was constructed as done for the case with penalty.
# In this case however, the initial wf is the same for BS and HF.
psi[1] = 1.0
circuit = QuantumCircuit(2)
circuit.x(0)

# BS
psi[2] = 1.0
circuit = QuantumCircuit(2)
circuit.x(1)

initial_layout = {circuit.qubits[0]: 57, circuit.qubits[1]: 58}
basis_gates = ['cx', 'rz', 'ry', 'rx', 'x']

pauli_strings = [
"II", "IX", "IY", "IZ",
"XI", "XX", "XY", "XZ",
"YI", "YX", "YY", "YZ",
"ZI", "ZX", "ZY", "ZZ"
]

# 2-qubit labels
h2_labels = ["II", "ZI", "IZ", "ZZ", "XX", "YY"]    #50(2.7)
# h2_labels = ["II", "XI", "XX", "ZZ", "IX", "YY"]    #with s2

# coeffs
h2_coeffs = [-0.3104, 0.1026, 0.0632, 0.3406, 0.1450, 0.1450]   #50(2.7)
# h2_coeffs = [-0.1511384611, -0.0196531851, -0.4997962036, -0.2100747487, -0.0196531851, -0.5000000000]   #50(2.7) with s2
# h2_coeffs = [-0.1253, 0.2374, -0.1603, 0.4892, 0.1050, 0.1050]   #20(1.2)

h2_sparse = SparsePauliOp(h2_labels, coeffs=h2_coeffs)
print('Initial Ordering')
print(h2_sparse)

def get_expectation_value(circuit: QuantumCircuit, H:SparsePauliOp):
    
    # # Using Statevector Estimator
    # ckt = circuit.copy(name="ckt")
    # estimator = StatevectorEstimator()
    # pub = (ckt, H)
    # job = estimator.run([pub])
    # result = job.result()[0]
    # return result.data.evs

    # Using Estimator
    # estimator = Estimator()
    # job = estimator.run([ckt], [H])
    # result = job.result().values[0]
    # return result

    # Using Runtime Estimator
    ckt = circuit.copy(name="ckt")
    backend = AerSimulator()
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3, basis_gates=basis_gates)
    isa_circuit = pm.run(ckt)
    isa_observable = H.apply_layout(isa_circuit.layout)
    estimator = Estimator(backend, options={"default_shots": int(1e4)})
    job = estimator.run([(isa_circuit, isa_observable)])
    pub_result = job.result()[0]
    expectation_value = pub_result.data.evs

    # # With Noise
    # ckt = circuit.copy(name="ckt")
    # pm = generate_preset_pass_manager(backend=real_backend, optimization_level=2, initial_layout=initial_layout, layout_method="trivial")
    # isa_circuit = pm.run(ckt)
    # isa_observable = H.apply_layout(isa_circuit.layout)
    # estimator = Estimator(mode=real_backend, options={"default_shots": int(1e4)})
    # job = estimator.run([(isa_circuit, isa_observable)])
    # pub_result = job.result()[0]
    # expectation_value = pub_result.data.evs

    gate_counts = isa_circuit.count_ops()
    circuit_depth = isa_circuit.depth()
    print(f"Gate counts: {gate_counts}")
    print(f"Circuit depth: {circuit_depth}")
    # isa_circuit.draw("mpl")
    # plt.savefig(f"isa_circuit.png")


    return expectation_value

def get_wf_fidelity(fci_state, testcircuit: QuantumCircuit):
    test_ckt = testcircuit.copy(name="ckt")
    test_state = Statevector(test_ckt)
    return None

def next_state(test_circuit, H: SparsePauliOp, db):
    test_ckt = test_circuit.copy(name="ckt")
    evolution_gate = PauliEvolutionGate(H, time=db)
    test_ckt.append(evolution_gate, [0, 1])
    return test_ckt

# Get the FCI energy and wavefunction for the wave function and Hamiltonian, store the wf in 'Um0'.
# May need to use davidson or other methods.

def QITE_step(circuit, h2_sparse):
    for ib in range (300):   #for ib in range (nsteps):

        sv = Statevector(circuit.copy())

        h2_sparse_matrix = h2_sparse.to_matrix(sparse=True)
        img_evolved_state = expm_multiply(-db * h2_sparse_matrix, sv.data)
        img_evolved_statevector = Statevector(img_evolved_state)
        delta_alpha = img_evolved_statevector - sv 
        # Get Pmu_psi which is equal to sigma_I^dagger|psi>
        Pmu_psi = np.zeros((16, 4), dtype=complex)
        # Apply each Pauli string and store the resulting statevector
        for i, pauli in enumerate(pauli_strings):
            # Create the Pauli operator and apply it to the state
            pauli_op = Pauli(pauli)
            new_state = sv.evolve(pauli_op)
            Pmu_psi[i] = new_state.data
        # Amat = 2*real(conj(Pmu_psi).Pmu_psi). It is equal to (S+S^T); S is a symmetric matrix.
        Amat = np.dot(np.conj(Pmu_psi),Pmu_psi.T)
        Amat = 2.0*np.real(Amat)

        # bvec = -2*Imag(Pmu_psi.conj(delta_alpha)). It is equal to -2*Im[<psi|sigma_I^dagger|delta_alpha>]
        bvec = np.dot(Pmu_psi,np.conj(delta_alpha))
        bvec = -2.0*np.imag(bvec)

        x = SciLA.lstsq(Amat,bvec)[0]
        xv[0] = x.copy()

        temp_hamilt = SparsePauliOp(pauli_strings, coeffs=-xv[0])
        real_evolution_gate = PauliEvolutionGate(temp_hamilt, db)

        # Get psi after real time evolution with these values of xv as |psi'> = e^{-i*xv*db}|psi>
        # Make psi = psi', Iterate
        circuit.append(real_evolution_gate, [0,1])

        # Get the expectation value 'ea' of Hamiltonian(H) wrt to the wave function(psi). This is the energy.
        # ea = <psi|H|psi>
        ea = get_expectation_value(circuit, h2_sparse)
        print(f"Expectation value: {ea}") 

xv.append(np.zeros(16))

# Initial expectation value
ea = get_expectation_value(circuit, h2_sparse)
print(f"Expectation value: {ea}")

QITE_step(circuit, h2_sparse)

# Add real time evolution. H|psi> = e^{-iHt}|psi>
# evolution_gate = PauliEvolutionGate(h2_sparse, time=db)
# circuit.append(evolution_gate, [0, 1])





