import socket, re, os

PREPROC = 'QUASI_RAW'

CONFIG = {
    'db': {
        'healthy': {
            'train': {'study': ['HCP', 'IXI'], 'diagnosis': 'control'},
            'validation': {'study': 'BIOBD', 'diagnosis': 'control'},
            'test': {'study': ['BSNIP'], 'diagnosis': ['control']},
            'dx_labels_mapping': {'control': 0}

            },
        'scz_kfolds': {
            'train': {'study': ['HCP', 'IXI', 'SCHIZCONNECT-VIP', 'PRAGUE'],
                  'diagnosis': ['control', 'FEP', 'schizophrenia']},
            'test': {'study': ['BSNIP'],
                 'diagnosis': ['control', 'schizophrenia']},
            'dx_labels_mapping': {'control': 0, 'FEP': 1, 'schizophrenia': 1}
            },
        'tiny_scz_kfolds': {
            'train': {'study': ['SCHIZCONNECT-VIP'],
                      'diagnosis': ['control', 'schizophrenia']},
            'test': {'study': ['BSNIP'],
                     'diagnosis': ['control', 'schizophrenia']},
            'dx_labels_mapping': {'control': 0, 'FEP': 1, 'schizophrenia': 1}
        },
        'bipolar_kfolds' :{
            'train': {
                'study': ['BIOBD'], 'diagnosis': ['control', 'bipolar disorder']
            },
            'test': {
                'study': ['BSNIP'], 'diagnosis': ['control', 'psychotic bipolar disorder']
            },
            'dx_labels_mapping': {'control': 0, 'bipolar disorder': 1, 'psychotic bipolar disorder': 1}
        },
        'bipolar_scz_kfolds':{
            'train': {
                'study': ['BIOBD', 'SCHIZCONNECT-VIP'],
                'diagnosis': ['control', 'bipolar disorder', 'schizophrenia']
            },
            'test': {
                'study': ['BSNIP'], 'diagnosis': ['control', 'psychotic bipolar disorder', 'schizophrenia']
            },
            'dx_labels_mapping': {'control': 0, 'FEP': 1, 'schizophrenia': 1,
                               'bipolar disorder': 2, 'psychotic bipolar disorder': 2}
        },
        'big_healthy': {
            'train': {'study': ['OASIS3', 'CoRR', 'HCP', 'ABIDE1', 'GSP', 'RBP', 'ABIDE2', 'IXI', 'LOCALIZER',
                                'MPI-LEIPZIG', 'ICBM', 'NPC', 'NAR'], 'diagnosis': 'control'},
            'validation': {'study': 'BIOBD', 'diagnosis': 'control'},
            'test': {'study': 'BSNIP', 'diagnosis': 'control'},
            'dx_labels_mapping': {"control": 0}
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


ALL_DATASETS = [('hcp_t1mri_{preproc}_{data}.npy', 'hcp_t1mri_{preproc}_participants.csv'),
                ('ixi_t1mri_{preproc}_{data}.npy', 'ixi_t1mri_{preproc}_participants.csv'),
                ('npc_t1mri_{preproc}_{data}.npy', 'npc_t1mri_{preproc}_participants.csv'),
                ('nar_t1mri_{preproc}_{data}.npy', 'nar_t1mri_{preproc}_participants.csv'),
                ('rbp_t1mri_{preproc}_{data}.npy', 'rbp_t1mri_{preproc}_participants.csv'),
                ('oasis3_t1mri_{preproc}_{data}.npy', 'oasis3_t1mri_{preproc}_participants.csv'),
                ('gsp_t1mri_{preproc}_{data}.npy', 'gsp_t1mri_{preproc}_participants.csv'),
                ('icbm_t1mri_{preproc}_{data}.npy', 'icbm_t1mri_{preproc}_participants.csv'),
                ('abide1_t1mri_{preproc}_{data}.npy', 'abide1_t1mri_{preproc}_participants.csv'),
                ('abide2_t1mri_{preproc}_{data}.npy', 'abide2_t1mri_{preproc}_participants.csv'),
                ('localizer_t1mri_{preproc}_{data}.npy', 'localizer_t1mri_{preproc}_participants.csv'),
                ('mpi-leipzig_t1mri_{preproc}_{data}.npy', 'mpi-leipzig_t1mri_{preproc}_participants.csv'),
                ('corr_t1mri_{preproc}_{data}.npy', 'corr_t1mri_{preproc}_participants.csv'),
                ## Datasets with scz
                ('candi_t1mri_{preproc}_{data}.npy', 'candi_t1mri_{preproc}_participants.csv'),
                ('cnp_t1mri_{preproc}_{data}.npy', 'cnp_t1mri_{preproc}_participants.csv'),
                ('biobd_t1mri_{preproc}_{data}.npy', 'biobd_t1mri_{preproc}_participants.csv'),
                ('bsnip_t1mri_{preproc}_{data}.npy', 'bsnip_t1mri_{preproc}_participants.csv'),
                ('schizconnect-vip_t1mri_{preproc}_{data}.npy', 'schizconnect-vip_t1mri_{preproc}_participants.csv'),
                ]


# The paths depend on the platform/server
if socket.gethostname() == 'kraken':
    root = '/home_local/bd261576'
    ## Dataset used for benchmark
    #CONFIG['cat12']['input_path'] = '/home_local/bd261576/all_t1mri_mwp1_gs-raw_data32_tocheck.npy'
    #CONFIG['cat12']['metadata_path'] = '/home_local/bd261576/all_t1mri_mwp1_participants.tsv'
    #CONFIG['quasi_raw']['input_path'] = '/home_local/bd261576/all_t1mri_quasi_raw_data32_1.5mm_skimage.npy'
    #CONFIG['quasi_raw']['metadata_path'] = '/home_local/bd261576/all_t1mri_quasi_raw_1.5mm_participants.tsv'

    ## All datasets
    CONFIG['cat12']['input_path'] = [
        os.path.join(root, 'all_datasets/cat12/{dataset}'.format(dataset=d[0].format(preproc="mwp1_gs-raw",
                                                                                     data="data64")))
        for d in ALL_DATASETS
    ]
    CONFIG['cat12']['metadata_path'] = [
        os.path.join(root, 'all_datasets/cat12/{phenotype}'.format(phenotype=d[1].format(preproc="mwp1")))
        for d in ALL_DATASETS
    ]
    CONFIG['cat12']['input_path_copy'] = [
        os.path.join(root, 'all_datasets_copy/cat12/{dataset}'.format(dataset=d[0].format(preproc="mwp1_gs-raw",
                                                                                          data="data64")))
        for d in ALL_DATASETS
    ]
    CONFIG['cat12']['metadata_path_copy'] = [
        os.path.join(root, 'all_datasets_copy/cat12/{phenotype}'.format(phenotype=d[1].format(preproc="mwp1")))
        for d in ALL_DATASETS
    ]
    CONFIG['quasi_raw']['input_path'] = [
        os.path.join(root, 'all_datasets/quasi_raw/{dataset}'.format(dataset=d[0].format(preproc="quasi_raw",
                                                                                         data="data32_1.5mm_skimage")))
        for d in ALL_DATASETS
    ]
    CONFIG['quasi_raw']['metadata_path'] = [
        os.path.join(root, 'all_datasets/quasi_raw/{phenotype}'.format(phenotype=d[1].format(preproc="quasi_raw")))
        for d in ALL_DATASETS
    ]
    CONFIG['quasi_raw']['input_path_copy'] = [
        os.path.join(root, 'all_datasets_copy/quasi_raw/{dataset}'.format(dataset=d[0].format(preproc="quasi_raw",
                                                                                         data="data32_1.5mm_skimage")))
        for d in ALL_DATASETS
    ]
    CONFIG['quasi_raw']['metadata_path_copy'] = [
        os.path.join(root, 'all_datasets_copy/quasi_raw/{phenotype}'.format(phenotype=d[1].format(preproc="quasi_raw")))
        for d in ALL_DATASETS
    ]


elif re.search("is[0-9]{6}", socket.gethostname()) is not None or socket.gethostname()=='triscotte' or \
        socket.gethostname()=='benoit-pc':
    root = '/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/'
    ## Dataset used for benchmark
    # CONFIG['cat12']['input_path'] = os.path.join(root, 'cat12vbm/all_t1mri_mwp1_gs-raw_data32_tocheck.npy')
    # CONFIG['cat12']['metadata_path'] = os.path.join(root, 'cat12vbm/all_t1mri_mwp1_participants.tsv')
    # CONFIG['quasi_raw']['input_path'] = os.path.join(root, 'quasi_raw/all_t1mri_quasi_raw_data32_1.5mm_skimage.npy')
    # CONFIG['quasi_raw']['metadata_path'] = os.path.join(root, 'quasi_raw/all_t1mri_quasi_raw_participants.tsv')

    # All datasets
    CONFIG['cat12']['input_path'] = [
        os.path.join(root, 'cat12vbm/{dataset}'.format(dataset=d[0].format(preproc="mwp1_gs-raw",
                                                                           data="data64"))) for d in ALL_DATASETS
    ]
    CONFIG['cat12']['metadata_path'] = [
        os.path.join(root, 'cat12vbm/{phenotype}'.format(phenotype=d[1].format(preproc='mwp1'))) for d in ALL_DATASETS
    ]
    CONFIG['quasi_raw']['input_path'] = [
        os.path.join(root, 'quasi_raw/{dataset}'.format(dataset=d[0].format(preproc="quasi_raw",
                                                                            data="data32_1.5mm_skimage"))) for d in ALL_DATASETS
    ]
    CONFIG['quasi_raw']['metadata_path'] = [
        os.path.join(root, 'quasi_raw/{phenotype}'.format(phenotype=d[1].format(preproc="quasi_raw"))) for d in ALL_DATASETS
    ]

else: ## Jean-Zay Cluster
    root = '/gpfsscratch/rech/lac/uoz16vf/data/'
    ## Dataset used for benchmark
    #CONFIG['cat12']['input_path'] = os.path.join(root, 'all_t1mri_mwp1_gs-raw_data32_tocheck.npy')
    #CONFIG['cat12']['metadata_path'] = os.path.join(root, 'all_t1mri_mwp1_participants.tsv')
    #CONFIG['quasi_raw']['input_path'] = os.path.join(root, 'all_t1mri_quasi_raw_data32_1.5mm_skimage.npy')
    #CONFIG['quasi_raw']['metadata_path'] = os.path.join(root,'all_t1mri_quasi_raw_participants.tsv')

    ## All datasets
    CONFIG['cat12']['input_path'] = [
        os.path.join(root, d[0].format(preproc="mwp1_gs-raw", data="data64")) for d in ALL_DATASETS
    ]
    CONFIG['cat12']['metadata_path'] = [
        os.path.join(root, d[1].format(preproc="mwp1")) for d in ALL_DATASETS
    ]
    CONFIG['quasi_raw']['input_path'] = [
        os.path.join(root, d[0].format(preproc="quasi_raw", data="data32_1.5mm_skimage")) for d in ALL_DATASETS
    ]
    CONFIG['quasi_raw']['metadata_path'] = [
        os.path.join(root, d[1].format(preproc="quasi_raw")) for d in ALL_DATASETS
    ]

