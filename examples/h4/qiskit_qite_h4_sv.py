import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply
from scipy.sparse import csr_matrix
from scipy import linalg as SciLA
import matplotlib.pyplot as plt
import pyzx as zx
from itertools import product

from ...src.real_device import BatchedEstimator
from ...src.inputs import get_json_file_path

import qiskit
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector, Pauli, Operator
from qiskit.primitives import StatevectorEstimator, Sampler, Estimator
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.circuit.library import PauliEvolutionGate
from qiskit_aer import AerSimulator
from qiskit.synthesis.evolution import LieTrotter
from qiskit.synthesis import QDrift, SuzukiTrotter, MatrixExponential, LieTrotter, ProductFormula

from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2 as Estimator, QiskitRuntimeService, SamplerV2 as Sampler

QiskitRuntimeService.save_account(channel="ibm_quantum", token="29c5c66c72c7c204e6c3dcce7daed1eaae03173b21c44b0177a4bec47229770bd72a4dfcd70a5f88e1ba774bb4f200886d7f1bee51ad5fa8a85e30f58cda1bc5", overwrite=True, set_as_default=True)
service = QiskitRuntimeService(channel="ibm_quantum", instance="utokyo-kawasaki/keio-internal/keio-students")
real_backend = service.backend("ibm_kawasaki")

xv = []
db = 0.2
bmax = 20
nsteps = bmax/db
nspin = 6

distance = 2.7
basis = "sto-3g"
spin = 0
charge = 0

psi = np.zeros(2**nspin, dtype=complex)
psi[36] = 1.0

circuit = QuantumCircuit(6)
circuit.x(2)
circuit.x(5)

initial_layout = {circuit.qubits[0]: 57, circuit.qubits[1]: 58, circuit.qubits[2]: 59, circuit.qubits[3]: 60, circuit.qubits[4]: 61, circuit.qubits[5]: 62}
basis_gates = ['cx', 'rz', 'ry', 'rx', 'x']

pauli_operators = ['I', 'X', 'Y', 'Z']
pauli_strings = ["".join(p) for p in product(pauli_operators, repeat=6)]

# 6-qubit labels
h4_labels = ['IIIIII', 'XXXIII', 'XXIXII', 'XXIXXI', 'XXIXZI', 'XXIYYZ', 'XXIZYY', 'XXIIXX', 'XXIIZX', 'XXIIIX', 'XYYIII', 'XYYXYY', 'XYYYYX', 'XYYZII', 'XYYZXI', 'XYYZZI', 'XYYIXZ', 'XYYIZZ', 'XYYIIZ', 'XZXIII', 'XZIXII', 'XZIXXI', 'XZIXZI', 'XZIYYZ', 'XZIZYY', 'XZIIXX', 'XZIIZX', 'XZIIIX', 'XIXIII', 'XIIXII', 'XIIXXI', 'XIIXZI', 'XIIYYZ', 'XIIZYY', 'XIIIXX', 'XIIIZX', 'XIIIIX', 'YXYIII', 'YYXIII', 'YYXXYY', 'YYXYYX', 'YYXZII', 'YYXZXI', 'YYXZZI', 'YYXIXZ', 'YYXIZZ', 'YYXIIZ', 'YYZXII', 'YYZXXI', 'YYZXZI', 'YYZYYZ', 'YYZZYY', 'YYZIXX', 'YYZIZX', 'YYZIIX', 'YZYIII', 'ZIIIII', 'ZXIIII', 'ZXZIII', 'ZXIXYY', 'ZXIYYX', 'ZXIZII', 'ZXIZXI', 'ZXIZZI', 'ZXIIXZ', 'ZXIIZZ', 'ZXIIIZ', 'ZYYXII', 'ZYYXXI', 'ZYYXZI', 'ZYYYYZ', 'ZYYZYY', 'ZYYIXX', 'ZYYIZX', 'ZYYIIX', 'ZZIIII', 'ZZZIII', 'ZZIXYY', 'ZZIYYX', 'ZZIZII', 'ZZIZXI', 'ZZIZZI', 'ZZIIXZ', 'ZZIIZZ', 'ZZIIIZ', 'ZIZIII', 'ZIIXYY', 'ZIIYYX', 'ZIIZII', 'ZIIZXI', 'ZIIZZI', 'ZIIIXZ', 'ZIIIZZ', 'ZIIIIZ', 'IXIIII', 'IXXXII', 'IXXXXI', 'IXXXZI', 'IXXYYZ', 'IXXZYY', 'IXXIXX', 'IXXIZX', 'IXXIIX', 'IXZIII', 'IXZXYY', 'IXZYYX', 'IXZZII', 'IXZZXI', 'IXZZZI', 'IXZIXZ', 'IXZIZZ', 'IXZIIZ', 'IZIIII', 'IZXXII', 'IZXXXI', 'IZXXZI', 'IZXYYZ', 'IZXZYY', 'IZXIXX', 'IZXIZX', 'IZXIIX', 'IZZIII', 'IZZXYY', 'IZZYYX', 'IZZZII', 'IZZZXI', 'IZZZZI', 'IZZIXZ', 'IZZIZZ', 'IZZIIZ', 'IIXXII', 'IIXXXI', 'IIXXZI', 'IIXYYZ', 'IIXZYY', 'IIXIXX', 'IIXIZX', 'IIXIIX', 'IIZIII', 'IIZXYY', 'IIZYYX', 'IIZZII', 'IIZZXI', 'IIZZZI', 'IIZIXZ', 'IIZIZZ', 'IIZIIZ', 'IIIXXX', 'IIIXYY', 'IIIXZX', 'IIIXIX', 'IIIYXY', 'IIIYYX', 'IIIYZY', 'IIIZII', 'IIIZXI', 'IIIZXZ', 'IIIZZI', 'IIIZZZ', 'IIIZIZ', 'IIIIXI', 'IIIIXZ', 'IIIIZI', 'IIIIZZ', 'IIIIIZ']

# coeffs
h4_coeffs = [-1.0554319189, 0.005687376, -0.0075995998, 0.03083671315, 0.0075995998, 0.03083671315, 0.02598776945, 0.02598776945, -0.0075572992, 0.0075572992, 0.0052721056, 0.03194549195, -0.03194549195, 0.0144043803, 0.03225585715, -0.00435703135, -0.03225585715, 0.0149395093, -0.0044758372, 0.0196517029, -0.04116096325, 0.0075995998, 0.04116096325, 0.0075995998, -0.01742661745, -0.01742661745, -0.0420817086, 0.0420817086, -0.0321878783, 0.04116096325, -0.0075995998, -0.04116096325, -0.0075995998, 0.01742661745, 0.01742661745, 0.0420817086, -0.0420817086, 0.005687376, -0.0052721056, -0.03194549195, 0.03194549195, -0.0144043803, -0.03225585715, 0.00435703135, 0.03225585715, -0.0149395093, 0.0044758372, -0.0075995998, 0.03083671315, 0.0075995998, 0.03083671315, 0.02598776945, 0.02598776945, -0.0075572992, 0.0075572992, -0.0125361754, 0.0816198499, -0.0065345441, 0.0064257161, 0.03225585715, -0.03225585715, 0.01495995835, 0.0329940343, -0.0037713819, -0.0329940343, 0.01570703925, -0.0044085394, 0.01742661745, 0.02598776945, -0.01742661745, 0.02598776945, 0.03639776355, 0.03639776355, 0.01819461405, -0.01819461405, 0.0563406733, 0.10700286605, -0.00435703135, 0.00435703135, 0.0797766048, -0.0037713819, 0.0830856456, 0.0037713819, 0.08260395075, 0.08374969635, 0.09820093645, 0.0144043803, -0.0144043803, 0.08762026935, 0.01495995835, 0.0797766048, -0.01495995835, 0.0904876464, 0.08053651195, -0.0064257161, 0.01742661745, 0.02598776945, -0.01742661745, 0.02598776945, 0.03639776355, 0.03639776355, 0.01819461405, -0.01819461405, 0.0065345441, -0.03225585715, 0.03225585715, -0.01495995835, -0.0329940343, 0.0037713819, 0.0329940343, -0.01570703925, 0.0044085394, 0.07858221425, 0.0420817086, -0.0075572992, -0.0420817086, -0.0075572992, 0.01819461405, 0.01819461405, 0.0437092569, -0.0437092569, -0.0169452601, 0.0149395093, -0.0149395093, 0.0904876464, 0.01570703925, 0.08260395075, -0.01570703925, 0.0945051958, 0.0836758296, -0.0420817086, 0.0075572992, 0.0420817086, 0.0075572992, -0.01819461405, -0.01819461405, -0.0437092569, 0.0437092569, 0.0200785332, -0.0044758372, 0.0044758372, 0.08053651195, -0.0044085394, 0.08374969635, 0.0044085394, 0.0836758296, 0.08535140905, 0.005687376, 0.0052721056, 0.0196517029, -0.0321878783, 0.005687376, -0.0052721056, -0.0125361754, 0.0816198499, -0.0065345441, 0.0064257161, 0.0563406733, 0.10700286605, 0.09820093645, -0.0064257161, 0.0065345441, 0.07858221425, -0.0169452601, 0.0200785332] 

h4_sparse = SparsePauliOp(h4_labels, coeffs=h4_coeffs)
print('Initial Ordering')
print(h4_sparse)

be = BatchedEstimator(backend_name="ibm_kawasaki")

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

    # # Using Runtime Estimator
    # ckt = circuit.copy(name="ckt")
    # backend = AerSimulator()
    # pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    # isa_circuit = pm.run(ckt)
    # isa_observable = H.apply_layout(isa_circuit.layout)
    # estimator = Estimator(backend, options={"default_shots": int(1e4)})
    # job = estimator.run([(isa_circuit, isa_observable)])
    # pub_result = job.result()[0]
    # expectation_value = pub_result.data.evs

    # With Noise
    ckt = circuit.copy(name="ckt")
    # aer = AerSimulator()
    pm = generate_preset_pass_manager(backend=real_backend, optimization_level=2, initial_layout=initial_layout, layout_method="trivial")
    isa_circuit = pm.run(ckt)
    isa_observable = H.apply_layout(isa_circuit.layout)
    estimator = Estimator(mode=real_backend, options={"default_shots": int(1e4)})
    job = estimator.run([(isa_circuit, isa_observable)])
    pub_result = job.result()[0]
    expectation_value = pub_result.data.evs
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

def QITE_step(circuit, h4_sparse):
# For every iteration
    for ib in range (4):   #for ib in range (nsteps):
        sv = Statevector(circuit.copy())

        h4_sparse_matrix = h4_sparse.to_matrix(sparse=True)
        img_evolved_state = expm_multiply(-db * h4_sparse_matrix, sv.data)
        img_evolved_statevector = Statevector(img_evolved_state)
        delta_alpha = img_evolved_statevector - sv 
        # Get Pmu_psi which is equal to sigma_I^dagger|psi>
        Pmu_psi = np.zeros((4096, 64), dtype=complex)
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
        real_evolution_gate = PauliEvolutionGate(temp_hamilt, db, synthesis=QDrift(reps=200))
        circuit.append(real_evolution_gate, [0,1,2,3,4,5])

        ckt = circuit.copy(name="ckt")
        pm = generate_preset_pass_manager(backend=real_backend, optimization_level=3, layout_method="sabre")
        isa_circuit = pm.run(ckt)
        # # print(isa_circuit.draw())
        # print(f"Depth: {isa_circuit.depth()}")
        isa_observable = h4_sparse.apply_layout(isa_circuit.layout)
        be.add_job(circuit=isa_circuit, operator=isa_observable)

    be.run_estimations(save_job_id_path=get_json_file_path())

        

xv.append(np.zeros(4096))

# Initial expectation value
# ea = get_expectation_value(circuit, h4_sparse)
# print(f"Expectation value: {ea}")

QITE_step(circuit, h4_sparse)

# Add real time evolution. H|psi> = e^{-iHt}|psi>
# evolution_gate = PauliEvolutionGate(h4_sparse, time=db)
# circuit.append(evolution_gate, [0, 1])





