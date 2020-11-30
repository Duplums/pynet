import socket, re, os

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
        'big_healthy': {
            'train': {'study': ['OASIS3', 'CoRR', 'HCP', 'ABIDE1', 'GSP', 'RBP', 'ABIDE2', 'IXI', 'LOCALIZER',
                                'MPI-LEIPZIG', 'ICBM', 'NPC', 'NAR'], 'diagnosis': 'control'},
            'validation': {'study': 'BIOBD', 'diagnosis': 'control'},
            'test': {'study': 'BSNIP', 'diagnosis': 'control'}
        }
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
ALL_DATASETS = [('hcp_t1mri_mwp1_gs-raw_data64.npy', 'hcp_t1mri_mwp1_participants.csv'),
                ('ixi_t1mri_mwp1_gs-raw_data64.npy', 'ixi_t1mri_mwp1_participants.csv'),
                ('npc_t1mri_mwp1_gs-raw_data64.npy', 'npc_t1mri_mwp1_participants.csv'),
                ('nar_t1mri_mwp1_gs-raw_data64.npy', 'nar_t1mri_mwp1_participants.csv'),
                ('rbp_t1mri_mwp1_gs-raw_data64.npy', 'rbp_t1mri_mwp1_participants.csv'),
                ('oasis3_t1mri_mwp1_gs-raw_data64.npy', 'oasis3_t1mri_mwp1_participants.csv'),
                ('gsp_t1mri_mwp1_gs-raw_data64.npy', 'gsp_t1mri_mwp1_participants.csv'),
                ('icbm_t1mri_mwp1_gs-raw_data64.npy', 'icbm_t1mri_mwp1_participants.csv'),
                ('abide1_t1mri_mwp1_gs-raw_data64.npy', 'abide1_t1mri_mwp1_participants.csv'),
                ('abide2_t1mri_mwp1_gs-raw_data64.npy', 'abide2_t1mri_mwp1_participants.csv'),
                ('localizer_t1mri_mwp1_gs-raw_data64.npy', 'localizer_t1mri_mwp1_participants.csv'),
                ('mpi-leipzig_t1mri_mwp1_gs-raw_data64.npy', 'mpi-leipzig_t1mri_mwp1_participants.csv'),
                ('corr_t1mri_mwp1_gs-raw_data64.npy', 'corr_t1mri_mwp1_participants.csv'),
                ## Datasets with scz
                ('candi_t1mri_mwp1_gs-raw_data64.npy', 'candi_t1mri_mwp1_participants.csv'),
                ('cnp_t1mri_mwp1_gs-raw_data64.npy', 'cnp_t1mri_mwp1_participants.csv'),
                ('biobd_t1mri_mwp1_gs-raw_data64.npy', 'biobd_t1mri_mwp1_participants.csv'),
                ('bsnip_t1mri_mwp1_gs-raw_data64.npy', 'bsnip_t1mri_mwp1_participants.csv'),
                ('schizconnect-vip_t1mri_mwp1_gs-raw_data64.npy', 'schizconnect-vip_t1mri_mwp1_participants.csv'),
                ]

# The paths depend on the platform/server
if socket.gethostname() == 'kraken':
    root = '/home_local/bd261576'
    ## Dataset used for benchmark
    #CONFIG['cat12']['input_path'] = '/home_local/bd261576/all_t1mri_mwp1_gs-raw_data32_tocheck.npy'
    #CONFIG['cat12']['metadata_path'] = '/home_local/bd261576/all_t1mri_mwp1_participants.tsv'
    CONFIG['quasi_raw']['input_path'] = '/home_local/bd261576/all_t1mri_quasi_raw_data32_1.5mm_skimage.npy'
    CONFIG['quasi_raw']['metadata_path'] = '/home_local/bd261576/all_t1mri_quasi_raw_1.5mm_participants.tsv'

    ## All datasets
    CONFIG['cat12']['input_path'] = [
        os.path.join(root, 'all_datasets/{dataset}'.format(dataset=d[0])) for d in ALL_DATASETS
    ]
    CONFIG['cat12']['metadata_path'] = [
        os.path.join(root, 'all_datasets/{phenotype}'.format(phenotype=d[1])) for d in ALL_DATASETS
    ]
    CONFIG['cat12']['input_path_copy'] = [
        os.path.join(root, 'all_datasets_copy/{dataset}'.format(dataset=d[0])) for d in ALL_DATASETS
    ]
    CONFIG['cat12']['metadata_path_copy'] = [
        os.path.join(root, 'all_datasets_copy/{phenotype}'.format(phenotype=d[1])) for d in ALL_DATASETS
    ]


elif re.search("is[0-9]{6}", socket.gethostname()) is not None or socket.gethostname()=='triscotte' or \
        socket.gethostname()=='benoit-pc':
    root = '/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/'
    ## Dataset used for benchmark
    # CONFIG['cat12']['input_path'] = os.path.join(root, 'cat12vbm/all_t1mri_mwp1_gs-raw_data32_tocheck.npy')
    # CONFIG['cat12']['metadata_path'] = os.path.join(root, 'cat12vbm/all_t1mri_mwp1_participants.tsv')
    CONFIG['quasi_raw']['input_path'] = os.path.join(root, 'quasi_raw/all_t1mri_quasi_raw_data32_1.5mm_skimage.npy')
    CONFIG['quasi_raw']['metadata_path'] = os.path.join(root, 'quasi_raw/all_t1mri_quasi_raw_participants.tsv')

    # All datasets
    CONFIG['cat12']['input_path'] = [
        os.path.join(root, 'cat12vbm/{dataset}'.format(dataset=d[0])) for d in ALL_DATASETS
    ]
    CONFIG['cat12']['metadata_path'] = [
        os.path.join(root, 'cat12vbm/{phenotype}'.format(phenotype=d[1])) for d in ALL_DATASETS
    ]

else: ## Jean-Zay Cluster
    root = '/gpfsscratch/rech/lac/uoz16vf/data/'
    ## Dataset used for benchmark
    #CONFIG['cat12']['input_path'] = os.path.join(root, 'all_t1mri_mwp1_gs-raw_data32_tocheck.npy')
    #CONFIG['cat12']['metadata_path'] = os.path.join(root, 'all_t1mri_mwp1_participants.tsv')
    CONFIG['quasi_raw']['input_path'] = os.path.join(root, 'all_t1mri_quasi_raw_data32_1.5mm_skimage.npy')
    CONFIG['quasi_raw']['metadata_path'] = os.path.join(root,'all_t1mri_quasi_raw_participants.tsv')

    ## All datasets
    CONFIG['cat12']['input_path'] = [
        os.path.join(root, d[0]) for d in ALL_DATASETS
    ]
    CONFIG['cat12']['metadata_path'] = [
        os.path.join(root, d[1]) for d in ALL_DATASETS
    ]

