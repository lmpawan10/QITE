import fireopal as fo
from qiskit_ibm_runtime import QiskitRuntimeService
import matplotlib.pyplot as plt
import qctrlvisualizer as qv
from qiskit import QuantumCircuit, qasm3

import time
import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply, lsqr
from scipy.sparse import csr_matrix
from scipy import linalg as SciLA
import matplotlib.pyplot as plt
from itertools import product

from qiskit.circuit import QuantumCircuit, EquivalenceLibrary
from qiskit.quantum_info import SparsePauliOp, Statevector, Pauli
from qiskit.circuit.library import PauliEvolutionGate, XGate
from qiskit_aer import AerSimulator
from qiskit.synthesis import QDrift, SuzukiTrotter, MatrixExponential, LieTrotter, ProductFormula

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2 as Estimator, QiskitRuntimeService, SamplerV2 as Sampler

from inputs import get_backend, get_channel, get_ibm_token, get_instance, get_qctrl_api_key, get_hub, get_group, get_project


fo.authenticate_qctrl_account(api_key=get_qctrl_api_key())

hub = get_hub()
group = get_group()
project = get_project()
token = get_ibm_token() 

credentials = fo.credentials.make_credentials_for_ibmq(
    token=token, hub=hub, group=group, project=project
)

QiskitRuntimeService.save_account(channel=get_channel(), token=get_ibm_token(), overwrite=True, set_as_default=True)
service = QiskitRuntimeService(channel=get_channel(), instance=get_instance())
real_backend = service.backend(get_backend())

jobs = []

xv = []
db = 0.2
bmax = 30
nsteps = bmax/db
nspin = 6
shot_count = 2000
backend_name = get_backend()
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

# no s2-1.99 for hf
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

# s2-1.99 for bs
# p4_labels = ['IIIIII', 'XXZIII', 'XXIXXI', 'XXIYYZ', 'XXIZZI', 'XXIIIZ', 'XYYXII', 'XYYXYY', 'XYYXZI', 'XYYYYX', 'XYYZXI', 'XYYIXZ', 'XYYIZX', 'XYYIIX', 'XZXIII', 'XZIXII', 'XZIXYY', 'XZIXZI', 'XZIYYX', 'XZIZXI', 'XZIIXZ', 'XZIIZX', 'XZIIIX', 'XIXIII', 'XIIXII', 'XIIXYY', 'XIIXZI', 'XIIYYX', 'XIIZXI', 'XIIIXZ', 'XIIIZX', 'XIIIIX', 'YYIIII', 'YYXXII', 'YYXXYY', 'YYXXZI', 'YYXYYX', 'YYXZXI', 'YYXIXZ', 'YYXIZX', 'YYXIIX', 'YYZXXI', 'YYZYYZ', 'YYZZZI', 'YYZIIZ', 'YZYIII', 'ZIIIII', 'ZXXIII', 'ZXIXII', 'ZXIXYY', 'ZXIXZI', 'ZXIYYX', 'ZXIZXI', 'ZXIIXZ', 'ZXIIZX', 'ZXIIIX', 'ZYYIII', 'ZYYZII', 'ZYYZYY', 'ZYYZZI', 'ZYYIXX', 'ZYYIZZ', 'ZYYIIZ', 'ZZIIII', 'ZZZIII', 'ZZIXXI', 'ZZIYYZ', 'ZZIZII', 'ZZIZYY', 'ZZIZZI', 'ZZIIXX', 'ZZIIZZ', 'ZZIIIZ', 'ZIZIII', 'ZIIZII', 'ZIIZYY', 'ZIIZZI', 'ZIIIXX', 'ZIIIZZ', 'ZIIIIZ', 'IXXIII', 'IXXZII', 'IXXZYY', 'IXXZZI', 'IXXIXX', 'IXXIZZ', 'IXXIIZ', 'IXZXII', 'IXZXYY', 'IXZXZI', 'IXZYYX', 'IXZZXI', 'IXZIXZ', 'IXZIZX', 'IXZIIX', 'IYYIII', 'IZIIII', 'IZXXII', 'IZXXYY', 'IZXXZI', 'IZXYYX', 'IZXZXI', 'IZXIXZ', 'IZXIZX', 'IZXIIX', 'IZZIII', 'IZZZII', 'IZZZYY', 'IZZZZI', 'IZZIXX', 'IZZIZZ', 'IZZIIZ', 'IIXXII', 'IIXXYY', 'IIXXZI', 'IIXYYX', 'IIXZXI', 'IIXIXZ', 'IIXIZX', 'IIXIIX', 'IIZIII', 'IIZXXI', 'IIZYYZ', 'IIZZII', 'IIZZYY', 'IIZZZI', 'IIZIXX', 'IIZIZZ', 'IIZIIZ', 'IIIXXZ', 'IIIXZX', 'IIIXIX', 'IIIYYI', 'IIIYZY', 'IIIZII', 'IIIZXX', 'IIIZYY', 'IIIZZI', 'IIIZZZ', 'IIIZIZ', 'IIIIXX', 'IIIIYY', 'IIIIZI', 'IIIIZZ', 'IIIIIZ']
# p4_coeffs = [-0.36556873325, -0.0277470173, 0.0197244159, 0.0197244159, -0.0198809362, 0.0198809362, 7.54648e-05, 0.03407250695, -7.54648e-05, -0.03407250695, 6.76934e-05, -6.76934e-05, -0.03375444485, 0.03375444485, -0.0002707736, -0.03407250695, -7.54648e-05, 0.03407250695, 7.54648e-05, -0.03375444485, 0.03375444485, 6.76934e-05, -6.76934e-05, 0.0001353868, 0.03407250695, 7.54648e-05, -0.03407250695, -7.54648e-05, 0.03375444485, -0.03375444485, -6.76934e-05, 6.76934e-05, -0.0277470173, -7.54648e-05, -0.03407250695, 7.54648e-05, 0.03407250695, -6.76934e-05, 6.76934e-05, 0.03375444485, -0.03375444485, 0.0197244159, 0.0197244159, -0.0198809362, 0.0198809362, -0.0001353868, 0.22477137715, 0.00013671065, 0.03375444485, 6.76934e-05, -0.03375444485, -6.76934e-05, 0.0370863534, -0.0370863534, -7.50028e-05, 7.50028e-05, -0.00096004455, -1.751835e-05, 0.0024818342, -1.311455e-05, 0.0024818342, -3.7614e-06, -1.311455e-05, -0.00410000295, 0.21112735265, -0.0198809362, -0.0198809362, 0.12628118985, -1.311455e-05, 0.1460621743, -1.311455e-05, 0.12921698775, 0.10596090855, 0.18433931725, 0.12985557505, -1.751835e-05, 0.12628118985, -1.751835e-05, 0.1273726942, 0.12628118985, -0.00096004455, -1.751835e-05, 0.0024818342, -1.311455e-05, 0.0024818342, -3.7614e-06, -1.311455e-05, -0.03375444485, -6.76934e-05, 0.03375444485, 6.76934e-05, -0.0370863534, 0.0370863534, 7.50028e-05, -7.50028e-05, 0.00013671065, 0.18433931725, -6.76934e-05, -0.03375444485, 6.76934e-05, 0.03375444485, -7.50028e-05, 7.50028e-05, 0.0370863534, -0.0370863534, -0.27939958025, 0.1273726942, -3.7614e-06, 0.12921698775, -3.7614e-06, 0.1348752153, 0.12921698775, 6.76934e-05, 0.03375444485, -6.76934e-05, -0.03375444485, 7.50028e-05, -7.50028e-05, -0.0370863534, 0.0370863534, -0.00410000295, 0.0198809362, 0.0198809362, 0.12628118985, -1.311455e-05, 0.10596090855, -1.311455e-05, 0.12921698775, 0.1460621743, -0.0277470173, -0.0002707736, 0.0001353868, -0.0277470173, -0.0001353868, 0.22477137715, 0.00013671065, -0.00096004455, -0.00410000295, 0.21112735265, 0.18433931725, -0.00096004455, 0.00013671065, 0.18433931725, -0.27939958025, -0.00410000295]

p4_sparse = SparsePauliOp(p4_labels, coeffs=p4_coeffs)

basis_gates = ['cx', 'rz', 'ry', 'rx', 'x']

pauli_operators = ['I', 'X', 'Y', 'Z']
pauli_strings = ["".join(p) for p in product(pauli_operators, repeat=6)]

observables = [(str(pauli), float(coeff)) for pauli, coeff in zip(p4_sparse.paulis, p4_sparse.coeffs)]

def QITE_step(circuit, p4_sparse):

    for ib in range (10):

        sv = Statevector(circuit.copy())

        p4_sparse_matrix = p4_sparse.to_matrix(sparse=True)
        img_evolved_state = expm_multiply(-db * p4_sparse_matrix, sv.data)
        img_evolved_statevector = Statevector(img_evolved_state)
        delta_alpha = img_evolved_statevector - sv

        Pmu_psi = np.zeros((4096, 64), dtype=complex)

        for i, pauli in enumerate(pauli_strings):
            # pauli_op = Pauli(pauli)
            new_state = sv.evolve(Pauli(pauli))
            Pmu_psi[i] = new_state.data

        Amat = np.dot(np.conj(Pmu_psi),Pmu_psi.T)
        Amat = 2.0*np.real(Amat)

        # bvec = -2*Imag(Pmu_psi.conj(delta_alpha)). It is equal to -2*Im[<psi|sigma_I^dagger|delta_alpha>]
        bvec = np.dot(Pmu_psi,np.conj(delta_alpha))
        bvec = -2.0*np.imag(bvec)

        x = SciLA.lstsq(Amat,bvec)[0]
        xv[0] = x.copy()
        print(f"x: {xv[0]}")
        
        temp_hamilt = SparsePauliOp(pauli_strings, coeffs=-xv[0])

        real_evolution_gate = PauliEvolutionGate(temp_hamilt, db, synthesis=QDrift(reps=200))
        circuit.append(real_evolution_gate, [0,1,2,3,4,5])
        evolved_circuit = circuit.copy(name=f"evolved_{ib}")
        
        fire_opal_job = fo.estimate_expectation(
            circuits=[qasm3.dumps(evolved_circuit)],
            shot_count=shot_count,
            credentials=credentials,
            backend_name=backend_name,
            observables=observables,
        )

        print(np.sum(fire_opal_job.result()["expectation_values"]))

xv.append(np.zeros(4096))
print(f"p4_sparse: {p4_sparse}")

QITE_step(circuit, p4_sparse)
