
def get_instance():
    instance = "utokyo-kawasaki/keio-internal/keio-students"
    return instance

def get_backend():
    backend = "ibm_kawasaki"
    return backend

def get_hub():
    parts = get_instance().rstrip('/').split('/')
    hub = parts[0]
    return hub

def get_group():
    parts = get_instance().rstrip('/').split('/')
    group = parts[1]
    return group

def get_project():
    parts = get_instance().rstrip('/').split('/')
    project = parts[2]
    return project

def get_ibm_token():
    previous_token = "13f6a6ac4efbd97724577b01cc226230a105b50986fdc77471abb8cdf465af996c9da0a16896919091c166f5ba3d6e43572026fcef2c293c1bb6442f58f94ac4"
    token = "29c5c66c72c7c204e6c3dcce7daed1eaae03173b21c44b0177a4bec47229770bd72a4dfcd70a5f88e1ba774bb4f200886d7f1bee51ad5fa8a85e30f58cda1bc5"
    return token

def get_qctrl_api_key():
    previous_key = "Kf0AcFmSzoa3cwM5SXccQEGMZpLlqwGlepoGBHja2Pe2bTm3U5"
    api_key = "759Djq2K6r3JEsDfO5nYdasbyuXD0XOKjrD1xeqv9Zbl0icmzD"
    return api_key

def get_channel():
    channel = "ibm_quantum"
    return channel

def get_json_file_path():
    json_file_path = "/Users/pawanpoudel/Documents/College contents/Quantum Chemistry/QITE/QITE-master/code_v4/Qiskit_Implementation/job_id_qd.json"
    return json_file_path

def get_h2_hamiltonian_file_path():
    h2_hamiltonian_file_path = "/Users/pawanpoudel/Documents/College contents/Quantum Chemistry/QITE/QITE-master/code_v4/h2.dat"
    return h2_hamiltonian_file_path

def get_h2_s2_hamiltonian_file_path():
    h2_s2_hamiltonian_file_path = "/Users/pawanpoudel/Documents/College contents/Quantum Chemistry/QITE/QITE-master/code_v4/h2_s2.dat"
    return h2_s2_hamiltonian_file_path

def get_h2_bkt_s2_hamiltonian_file_path():
    h2_bkt_s2_hamiltonian_file_path = "/Users/pawanpoudel/Documents/College contents/Quantum Chemistry/QITE/QITE-master/code_v4/h2_bkt_s2.dat"
    return h2_bkt_s2_hamiltonian_file_path

def get_h2_s2_integral_files_path():
    h2_s2_integral_files_path = "/Users/pawanpoudel/Documents/College contents/Quantum Chemistry/QITE/QITE-master/numerics/h2/H2_STO-6G_Integrals/H2_R_UNOLoc_STO-6G_Int.txt"
    return h2_s2_integral_files_path

def get_4h_integral_files_path():
    h4_integral_files_path = "/Users/pawanpoudel/Documents/College contents/Quantum Chemistry/QITE/QITE-master/numerics/4h/4H_chain_R20_RHF_STO-3G_Int.txt"
    return h4_integral_files_path

def get_p4_s2_integral_files_path():
    p4_s2_integral_files_path = "/Users/pawanpoudel/Documents/College contents/Quantum Chemistry/QITE/4H_Cluster_Integrals/P4/P4_a02020_UNOMix_STO-3G_Int.txt"
    return p4_s2_integral_files_path

def get_n2_integral_files_path():
    n2_integral_files_path = "/Users/pawanpoudel/Documents/College contents/Quantum Chemistry/QITE/Integrals_N2_RHF/N2_R300_STO-3G_FC4_RHFInt.out"
    return n2_integral_files_path

def get_n2_bs2_integral_files_path():
    n2_bs2_integral_files_path = "/Users/pawanpoudel/Documents/College contents/Quantum Chemistry/QITE/Integrals_N2_BS2/N2_R300_STO-3G_UNOLoc2_Int.out"
    return n2_bs2_integral_files_path

def get_n2_bs3_integral_files_path():
    n2_bs3_integral_files_path = "/Users/pawanpoudel/Documents/College contents/Quantum Chemistry/QITE/Integrals_N2_BS3/N2_R300_STO-3G_UNOLoc3_Int.out"
    return n2_bs3_integral_files_path