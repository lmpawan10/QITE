from qiskit_ibm_runtime import QiskitRuntimeService, Session
from real_device import BatchedEstimator
from inputs import get_backend, get_channel, get_ibm_token, get_instance, get_json_file_path
 
QiskitRuntimeService.save_account(channel=get_channel(), token=get_ibm_token(), overwrite=True, set_as_default=True)
service = QiskitRuntimeService(channel=get_channel(), instance=get_instance())
backend = service.backend(get_backend())
 
be = BatchedEstimator(backend_name=get_backend())
results_path = get_json_file_path()
results = be.retrieve_results(results_path)
print("Results:", results)