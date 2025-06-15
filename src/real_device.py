from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator, Session
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from typing import Optional, Tuple
import json

from inputs import get_backend, get_channel, get_ibm_token, get_instance


class BatchedEstimator:
    def __init__(self, backend_name=get_backend()):
        QiskitRuntimeService.save_account(channel=get_channel(), token=get_ibm_token(), overwrite=True, set_as_default=True)
        self.service = QiskitRuntimeService(channel=get_channel(), instance=get_instance())
        self.backend = self.service.backend(backend_name)
        self.jobs: list[Tuple[QuantumCircuit, SparsePauliOp]] = []
        self.job_id = None

    def add_job(self, circuit: QuantumCircuit, operator: SparsePauliOp):
        self.jobs.append((circuit, operator))

    def run_estimations(self, save_job_id_path: Optional[str] = None):
        """
        Submit the estimator job and save job ID (optional).
        """
        with Session(backend=self.backend) as session:
            estimator = Estimator(mode=session, 
                                  options={
                                            "default_shots": int(3e4), 
                                            "resilience_level": 0, 
                                            "dynamical_decoupling": 
                                            {
                                                "enable": True, 
                                                "sequence_type": "XY4"
                                                # "skip_reset_qubits": False
                                            },
                                            "resilience":
                                            {
                                                # "measure_mitigation": True,
                                                "zne_mitigation": True,
                                                "zne": {
                                                    # "amplifier": "gate_folding",
                                                    "noise_factors": [1, 3, 5],
                                                    "extrapolator": ["exponential", "linear"]
                                                }
                                                
                                            },
                                            "twirling":
                                            {
                                                "enable_gates": True,
                                                "num_randomizations": "auto"
                                                # "strategy": "all"
                                            }
                                            })
            job = estimator.run(self.jobs)
            self.job_id = job.job_id()

            if save_job_id_path:
                with open(save_job_id_path, "w") as f:
                    json.dump({"job_id": self.job_id}, f)

            print(f"Job submitted. ID: {self.job_id}")
            return self.job_id

    def retrieve_results(self, saved_job_id_path: Optional[str] = None):
        """
        Retrieve the results later from a saved job ID.
        """
        job_id = self.job_id

        if saved_job_id_path:
            with open(saved_job_id_path, "r") as f:
                job_id = json.load(f)["job_id"]
        else:
            job_id = self.job_id

        if not job_id:
            raise ValueError("No job ID provided or saved.")

        job = self.service.job(job_id)
        result = job.result()
        evs = [res.data.evs for res in result]
        return evs
