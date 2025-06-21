import time
import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply, lsqr
from scipy.sparse import csr_matrix
from scipy import linalg as SciLA
import matplotlib.pyplot as plt
from itertools import product

from .src.inputs import get_backend, get_channel, get_ibm_token, get_instance, get_json_file_path
from .src.real_device import BatchedEstimator

import qiskit
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector, Pauli
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.circuit.library import PauliEvolutionGate, XGate
from qiskit_aer import AerSimulator
from qiskit.synthesis import QDrift, SuzukiTrotter, MatrixExponential, LieTrotter, ProductFormula

from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2 as Estimator, QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_ibm_runtime.transpiler.passes.scheduling import PadDynamicalDecoupling, ALAPScheduleAnalysis

QiskitRuntimeService.save_account(channel=get_channel(), token=get_ibm_token(), overwrite=True, set_as_default=True)
service = QiskitRuntimeService(channel=get_channel(), instance=get_instance())
real_backend = service.backend(get_backend())

xv = []
db = 0.1
bmax = 30
nsteps = bmax/db
nspin = 6
distance = 2.7
basis = "sto-3g"
spin = 0
charge = 0

psi = np.zeros(2**nspin, dtype=complex)

# HF
psi[36] = 1.0
circuit = QuantumCircuit(6)
circuit.x(2)
circuit.x(5)

# # BS
# psi[38] = 1.0
# circuit = QuantumCircuit(6)
# circuit.x(1)
# circuit.x(2)
# circuit.x(5)

initial_layout = {circuit.qubits[0]: 57, circuit.qubits[1]: 58, circuit.qubits[2]: 59, circuit.qubits[3]: 60, circuit.qubits[4]: 61, circuit.qubits[5]: 62}
basis_gates = ['cx', 'rz', 'ry', 'rx', 'x']

# p4_labels = ['IIIIII', 'XIXXIX', 'XZYYIX', 'XYYZIX', 'XXXIIX', 'XYYXYY', 'XXYYYY', 'XIXZYY', 'XZXIYY', 'XIIXIZ', 'XIZXIZ', 'XXZIIZ', 'XXIIIZ', 'XIIIXI', 'XIIXII', 'XIZXII', 'XXZIII', 'XXIIII', 'YYYXXY', 'YXYYXY', 'YIXZXY', 'YZXIXY', 'YIXXZY', 'YZYYZY', 'YYYZZY', 'YXXIZY', 'YIIIYZ', 'ZIIIII', 'ZYYXIX', 'ZXYYIX', 'ZIXZIX', 'ZZXIIX', 'ZIXXYY', 'ZZYYYY', 'ZYYZYY', 'ZXXIYY', 'ZIIIIZ', 'ZIIIZZ', 'ZIIZIZ', 'ZIZZIZ', 'ZZZIIZ', 'ZZIIIZ', 'ZIIIZI', 'ZIIZII', 'ZIZZII', 'ZZZIII', 'ZZIIII', 'IIXXXX', 'IZYYXX', 'IYYZXX', 'IXXIXX', 'IYYXZX', 'IXYYZX', 'IIXZZX', 'IZXIZX', 'IIIIIZ', 'IIIXXZ', 'IIZXXZ', 'IXZIXZ', 'IXIIXZ', 'IIIIZZ', 'IIIZZZ', 'IIZZZZ', 'IZZIZZ', 'IZIIZZ', 'IIIXXI', 'IIZXXI', 'IXZIXI', 'IXIIXI', 'IIIIZI', 'IIIZZI', 'IIZZZI', 'IZZIZI', 'IZIIZI', 'IXIXII', 'IYZYII', 'IIIZII', 'IIZZII', 'IZZZII', 'IZIZII', 'IIZIII', 'IZZIII', 'IZIIII']
# p4_coeffs = [-0.370911, 0.019741, 0.019741, -0.019902, -0.019902, 0.034066, -0.034066, -0.033761, 0.033761, -0.034066, 0.034066, 0.033761, -0.033761, 0.027718, 0.034066, -0.034066, -0.033761, 0.033761, -0.034066, 0.034066, 0.033761, -0.033761, 0.019741, 0.019741, -0.019902, -0.019902, 0.027718, 0.224153, -0.033761, 0.033761, 0.037094, -0.037094, -0.019902, -0.019902, 0.020077, 0.020077, -0.003884, 0.210813, 0.126136, 0.128343, 0.129073, 0.123406, 0.184048, 0.129693, 0.126136, 0.127224, 0.126136, -0.019902, -0.019902, 0.020077, 0.020077, 0.033761, -0.033761, -0.037094, 0.037094, 0.184048, -0.033761, 0.033761, 0.037094, -0.037094, -0.277880, 0.127224, 0.129073, 0.134714, 0.129073, 0.033761, -0.033761, -0.037094, 0.037094, -0.003884, 0.126136, 0.123406, 0.129073, 0.128343, 0.027718, 0.027718, 0.224153, -0.003884, 0.210813, 0.184048, 0.184048, -0.277880, -0.003884]

# p4_labels = ['IIIIII', 'XIXXIX', 'XZYYIX', 'XYYZIX', 'XXXIIX', 'XYYXYY', 'XXYYYY', 'XIXZYY', 'XZXIYY', 'XIIXIZ', 'XIZXIZ', 'XXZIIZ', 'XXIIIZ', 'XIIIXI', 'XIIXII', 'XIZXII', 'XXZIII', 'XXIIII', 'YYYXXY', 'YXYYXY', 'YIXZXY', 'YZXIXY', 'YIXXZY', 'YZYYZY', 'YYYZZY', 'YXXIZY', 'YIIIYZ', 'ZIIIII', 'ZYYXIX', 'ZXYYIX', 'ZIXZIX', 'ZZXIIX', 'ZIXXYY', 'ZZYYYY', 'ZYYZYY', 'ZXXIYY', 'ZIIIIZ', 'ZIIIZZ', 'ZIIZIZ', 'ZIZZIZ', 'ZZZIIZ', 'ZZIIIZ', 'ZIIIZI', 'ZIIZII', 'ZIZZII', 'ZZZIII', 'ZZIIII', 'IIXXXX', 'IZYYXX', 'IYYZXX', 'IXXIXX', 'IYYXZX', 'IXYYZX', 'IIXZZX', 'IZXIZX', 'IIIIIZ', 'IIIXXZ', 'IIZXXZ', 'IXZIXZ', 'IXIIXZ', 'IIIIZZ', 'IIIZZZ', 'IIZZZZ', 'IZZIZZ', 'IZIIZZ', 'IIIXXI', 'IIZXXI', 'IXZIXI', 'IXIIXI', 'IIIIZI', 'IIIZZI', 'IIZZZI', 'IZZIZI', 'IZIIZI', 'IXIXII', 'IYZYII', 'IIIZII', 'IIZZII', 'IZZZII', 'IZIZII', 'IIZIII', 'IZZIII', 'IZIIII']
# p4_coeffs = [-0.3709108589500001, 0.019740609500000002, 0.019740609500000002, -0.0199020831, -0.0199020831, 0.03406610945, -0.03406610945, -0.0337612592, 0.0337612592, -0.03406610945, 0.03406610945, 0.0337612592, -0.0337612592, 0.027718352199999997, 0.03406610945, -0.03406610945, -0.0337612592, 0.0337612592, -0.03406610945, 0.03406610945, 0.0337612592, -0.0337612592, 0.019740609500000002, 0.019740609500000002, -0.0199020831, -0.0199020831, 0.027718352200000004, 0.2241531948000004, -0.0337612592, 0.0337612592, 0.03709430705, -0.03709430705, -0.0199020831, -0.0199020831, 0.02007680635, 0.02007680635, -0.0038839750499999603, 0.21081273514999999, 0.12613611505, 0.12834316575, 0.12907256025, 0.12340606295, 0.1840482588, 0.12969344345, 0.12613611505, 0.12722408805, 0.12613611505, -0.0199020831, -0.0199020831, 0.02007680635, 0.02007680635, 0.0337612592, -0.0337612592, -0.03709430705, 0.03709430705, 0.1840482588, -0.0337612592, 0.0337612592, 0.03709430705, -0.03709430705, -0.27788003180000004, 0.12722408805, 0.12907256025, 0.1347139307, 0.12907256025, 0.0337612592, -0.0337612592, -0.03709430705, 0.03709430705, -0.0038839750499998216, 0.12613611505, 0.12340606295, 0.12907256025, 0.12834316575, 0.027718352199999997, 0.027718352200000004, 0.2241531948000004, -0.0038839750499999603, 0.21081273514999999, 0.1840482588, 0.1840482588, -0.2778800318000001, -0.00388397504999978]

# P4 - 1.99 - no_s2
p4_labels = [   
    "IIIIII", 
    "XXIXXI",
    "XXIYYZ",
    "XXIZYY",
    "XXIIXX",
    "ZYYXXI",
    "ZYYYYZ",
    "YYZZYY",
    "YYZIXX",
    "IXXXXI",
    "IXXYYZ",
    "XYYZXI",
    "XYYIXZ",
    "YYXZXI",
    "YYXIXZ",
    "ZXIXYY",
    "ZXIYYX",
    "IXZXYY",
    "IXZYYX",
    "XZIXII",
    "XZIXZI",
    "XIIXII",
    "XIIXZI",
    "XIIIZX",
    "XIIIIX",
    "XYYXYY",
    "XYYYYX",
    "YYXXYY",
    "YYXYYX",
    "IIIYZY",
    "YZYIII",
    "XIXIII",
    "IIIXIX",
    "XZIIZX",
    "XZIIIX",
    "IIXXII",
    "IIXXZI",
    "IZXXII",
    "IZXXZI",
    "ZXIZXI",
    "ZXIIXZ",
    "IXZZXI",
    "IXZIXZ",
    "IXXZYY",
    "IXXIXX",
    "ZYYZYY",
    "ZYYIXX",
    "IZXIZX",
    "IZXIIX",
    "IIXIZX",
    "IIXIIX",
    "IIIXZX",
    "XZXIII",
    "YYZXXI",
    "YYZYYZ",
    "IZZIIZ",
    "IIZIZZ",
    "IZZIII",
    "IIIIZZ",
    "ZZIIII",
    "IIIZZI",
    "ZZIZII",
    "ZIIZZI",
    "ZZIIZZ",
    "IZZZZI",
    "IIIZIZ",
    "ZIZIII",
    "IZZZII",
    "ZIIIZZ",
    "ZIIIIZ",
    "IIZZII",
    "ZZIIIZ",
    "IIZZZI",
    "IIIZII",
    "ZIIIII",
    "ZZZIII",
    "IIIZZZ",
    "IZIIII",
    "IIIIZI",
    "IIZIII",
    "IIIIIZ",
    "ZZIZZI",
    "ZIIZII",
    "IIZIIZ",
    "IZZIZZ"
]
p4_coeffs = [
    -0.365568733,
    0.019724416,
    0.019724416,
    -0.019880936,
    -0.019880936,
    -0.019880936,
    -0.019880936,
    -0.019880936,
    -0.019880936,
    -0.019880936,
    -0.019880936,
    -0.033686752,
    0.033686752,
    0.033686752,
    -0.033686752,
    -0.033686752,
    0.033686752,
    0.033686752,
    -0.033686752,
    -0.034147972,
    0.034147972,
    0.034147972,
    -0.034147972,
    -0.033822138,
    0.033822138,
    0.033997042,
    -0.033997042,
    -0.033997042,
    0.033997042,
    0.02761163,
    0.02761163,
    0.027882404,
    0.027882404,
    0.033822138,
    -0.033822138,
    0.033822138,
    -0.033822138,
    -0.033822138,
    0.033822138,
    0.037011351,
    -0.037011351,
    -0.037011351,
    0.037011351,
    0.020050633,
    0.020050633,
    0.020050633,
    0.020050633,
    0.037161356,
    -0.037161356,
    -0.037161356,
    0.037161356,
    -0.000270774,
    -0.000270774,
    0.019724416,
    0.019724416,
    0.129213226,
    0.129213226,
    -0.279399581,
    -0.279399581,
    -0.003139958,
    -0.003139958,
    0.126298708,
    0.126298708,
    0.129220749,
    0.129220749,
    0.184476028,
    0.184476028,
    0.127372694,
    0.127372694,
    0.126263672,
    0.126263672,
    0.123529707,
    0.123529707,
    0.224771377,
    0.224771377,
    0.211127353,
    0.211127353,
    0.184202607,
    0.184202607,
    -0.005060048,
    -0.005060048,
    0.128519605,
    0.129855575,
    0.128467147,
    0.134875215
]

threshold = 1e-3
p4_sparse = SparsePauliOp(p4_labels, coeffs=p4_coeffs)
print('Initial Ordering')
print(p4_sparse)

pauli_operators = ['I', 'X', 'Y', 'Z']
pauli_strings = ["".join(p) for p in product(pauli_operators, repeat=6)]

be = BatchedEstimator(backend_name=get_backend())

def get_expectation_value(circuit: QuantumCircuit, H:SparsePauliOp):
    
    # # # Using Statevector Estimator
    # ckt = circuit.copy(name="ckt")
    # estimator = StatevectorEstimator()
    # pub = (ckt, H)
    # job = estimator.run([pub])
    # result = job.result()[0]
    # return result.data.evs

    # With Runtime Estimator
    # ckt = circuit.copy(name="ckt")
    # backend = AerSimulator()
    # pm = generate_preset_pass_manager(backend=backend, optimization_level=3, layout_method="sabre")
    # isa_circuit = pm.run(circuit)
    # isa_observable = H.apply_layout(isa_circuit.layout)
    # estimator = Estimator(backend, options={"default_shots": int(1e4)})
    # job = estimator.run([(isa_circuit, isa_observable)])
    # pub_result = job.result()[0]
    # expectation_value = pub_result.data.evs

    # With Noise
    ckt = circuit.copy(name="ckt")
    pm = generate_preset_pass_manager(backend=real_backend, optimization_level=3, layout_method="sabre")
    isa_circuit = pm.run(ckt)
    isa_observable = H.apply_layout(isa_circuit.layout)
    estimator = Estimator(mode=real_backend, options={"default_shots": int(1e4), "resilience_level": 2})
    job = estimator.run([(isa_circuit, isa_observable)])
    pub_result = job.result()[0]
    expectation_value = pub_result.data.evs


    gate_counts = isa_circuit.count_ops()
    circuit_depth = isa_circuit.depth()
    print(f"Gate counts: {gate_counts}")
    print(f"Circuit depth: {circuit_depth}")
    # isa_circuit.draw("mpl")
    # plt.savefig(f"p4-199-nos2-cx-rx-ry-rz-x-basis_sets.png", dpi=50)

    return expectation_value


def build_trotterized_evolution_circuit(circuit, pauli_labels, xv_list, t=1.0, r=10):
    """
    Builds a first-order Trotterized circuit approximating exp(-iHt)
    using gate-level simulation via PauliEvolutionGate.
    
    Parameters:
    - pauli_labels: list of Pauli strings (e.g. ["ZIII", "IZII", ...])
    - xv_list: list of coefficient arrays/lists for partial Hamiltonians
    - num_qubits: number of qubits
    - t: total time of evolution
    - r: number of Trotter steps
    """
    dt = t / r
    # circuit = QuantumCircuit(num_qubits)
    num_qubits = 6

    for _ in range(r):
        for xv in xv_list:
            coeffs = xv[0] if isinstance(xv, list) else xv
            sparse_op = SparsePauliOp(pauli_labels, coeffs=coeffs * dt)
            evo_gate = PauliEvolutionGate(sparse_op, time=1.0)  # time=1 because coeffs are scaled
            circuit.append(evo_gate, range(num_qubits))
    
    return circuit

def QITE_step(circuit, p4_sparse):
    # comp_circuit = circuit.copy()
    comp_circuit = QuantumCircuit(6)
    # State 
    comp_circuit.x(2)
    comp_circuit.x(5)
    for ib in range (8):   #for ib in range (nsteps):
        sv = Statevector(circuit.copy())

        p4_sparse_matrix = p4_sparse.to_matrix(sparse=True)
        img_evolved_state = expm_multiply(-db * p4_sparse_matrix, sv.data)
        img_evolved_statevector = Statevector(img_evolved_state)
        # img_evolved_statevector = Statevector([complex(round(a.real, 10), round(a.imag, 10)) for a in Statevector(img_evolved_state).data])
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

        start_time = time.time()
        # x = lsqr(Amat, bvec)[0] 
        x = SciLA.lstsq(Amat,bvec)[0]
        xv[0] = x.copy()

        temp_hamilt = SparsePauliOp(pauli_strings, coeffs=-xv[0])

        real_evolution_gate = PauliEvolutionGate(temp_hamilt, db, synthesis=QDrift(reps=200))

        circuit.append(real_evolution_gate, [0,1,2,3,4,5])
        
        ckt = circuit.copy(name="ckt")
        pm = generate_preset_pass_manager(backend=real_backend, optimization_level=3, layout_method="sabre")
        isa_circuit = pm.run(ckt)
        isa_observable = p4_sparse.apply_layout(isa_circuit.layout)
        be.add_job(circuit=isa_circuit, operator=isa_observable)

    be.run_estimations(save_job_id_path=get_json_file_path())

xv.append(np.zeros(4096))

# Initial expectation value
# Get the expectation value 'ea' of Hamiltonian(H) wrt to the wave function(psi). This is the energy.
# ea = <psi|H|psi>
# ea = get_expectation_value(circuit, p4_sparse)
# print(f"Expectation value: {ea}")

QITE_step(circuit, p4_sparse)
