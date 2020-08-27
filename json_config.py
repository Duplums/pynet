import socket, re

PREPROC = 'QUASI_RAW'

CONFIG = {
    'db': {
        'healthy': {
            'train': {'study': ['HCP', 'IXI'], 'diagnosis': 'control'},
            'validation': {'study': 'BIOBD', 'diagnosis': 'control'},
            'test': {'study': ['BSNIP'], 'diagnosis': ['control']}
            },
        'scz_kfolds': {
            'train': {'study': ['HCP', 'IXI', 'SCHIZCONNECT-VIP', 'PRAGUE'],
                  'diagnosis': ['control', 'FEP', 'schizophrenia']},
            'test': {'study': ['BSNIP'],
                 'diagnosis': ['control', 'schizophrenia']}
            },
        'tiny_scz_kfolds': {
            'train': {'study': ['SCHIZCONNECT-VIP'],
                      'diagnosis': ['control', 'schizophrenia']},
            'test': {'study': ['BSNIP'],
                     'diagnosis': ['control', 'schizophrenia']}
            },
        },

    'optimizer': {
        'Adam': {'weight_decay': 5e-5}
    },
    'scheduler': {
        'StepLR': {} # By default step_size = 10
    },
    'cat12': {},
    'quasi_raw': {}
}
# The paths depend on the platform/server

if socket.gethostname() == 'kraken':
    CONFIG['cat12']['input_path'] = '/home_local/bd261576/all_t1mri_mwp1_gs-raw_data32_tocheck.npy'
    CONFIG['cat12']['metadata_path'] = '/home_local/bd261576/all_t1mri_mwp1_participants.tsv'
    CONFIG['quasi_raw']['input_path'] = '/home_local/bd261576/all_t1mri_quasi_raw_data32.npy'
    CONFIG['quasi_raw']['metadata_path'] = '/home_local/bd261576/all_t1mri_quasi_raw_participants.tsv'

elif re.search("is[0-9]{6}", socket.gethostname()) is not None or socket.gethostname()=='triscotte':
    CONFIG['cat12']['input_path'] = '/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/' \
                                    'data/cat12vbm/all_t1mri_mwp1_gs-raw_data32_tocheck.npy'
    CONFIG['cat12']['metadata_path'] = '/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/' \
                                       'data/cat12vbm/all_t1mri_mwp1_participants.tsv'
    CONFIG['quasi_raw']['input_path'] = '/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/' \
                                        'data/quasi_raw/all_t1mri_quasi_raw_data32.npy'
    CONFIG['quasi_raw']['metadata_path'] = '/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/' \
                                           'data/quasi_raw/all_t1mri_quasi_raw_participants.tsv'
else:
    CONFIG['cat12']['input_path'] = '/gpfsscratch/rech/lac/uoz16vf/data/all_t1mri_mwp1_gs-raw_data32_tocheck.npy'
    CONFIG['cat12']['metadata_path'] = '/gpfsscratch/rech/lac/uoz16vf/data/all_t1mri_mwp1_participants.tsv'
    CONFIG['quasi_raw']['input_path'] = '/gpfsscratch/rech/lac/uoz16vf/data/all_t1mri_quasi_raw_data32.npy'
    CONFIG['quasi_raw']['metadata_path'] = '/gpfsscratch/rech/lac/uoz16vf/data/all_t1mri_quasi_raw_participants.tsv'
