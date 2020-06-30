CONFIG = {
    ## Specific to Kraken TODO: use the computer name to set this automatically
    # 'input_path': '/home_local/bd261576/all_t1mri_mwp1_gs-raw_data32_tocheck.npy',
    # 'metadata_path': '/home_local/bd261576/all_t1mri_mwp1_participants.tsv',
    'input_path': '/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/'
                  'data/cat12vbm/all_t1mri_mwp1_gs-raw_data32_tocheck.npy',
    'metadata_path': '/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/'
                     'data/cat12vbm/all_t1mri_mwp1_participants.tsv',
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
        'StepLR': {'step_size': 10}
    }
}
