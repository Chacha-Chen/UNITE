
DIC_STN_CONF = {
    'emb': 16,
    'heads': 8,
    'depth': 2,
    'seq_length': 200,
    'vocab_size': 31467, #26706,
    'num_classes': 2,
    'max_pool': True,
    'dropout': 0.1,
    'embed': False,
    'wide': False
}

DIC_STNB_CONF = {
    'emb': 16,
    'heads': 8,
    'depth': 2,
    'seq_length': 200,
    'vocab_size': 31467, #26706,
    'num_classes': 2,
    'max_pool': True,
    'dropout': 0.1,
    'embed': False,
    'wide': False
}



DIC_VGP_CONF = {
    'embedding_dim' : 8,
    'vocab_size' : 26706, ## TODO adhoc
    # vocab_size : 1800,
    'hidden_dim' : 2,
    'num_layers' : 2,
    'output_dim' : 2,
    'batch_size': 256
}


DIC_CTRANSFORMER_CONF = {
        'emb'         :16,
        'heads'       :8,
        'depth'       :2,
        'seq_length'  :200,
        'num_tokens'  :26706,
        'num_classes' :2,
        'max_pool'    :True,
        'dropout'       :0.0 ,
        'wide'          :False
}