from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# PARAMS
# -----------------------------------------------------------------------------
_C.PARAMS = CN()
# Parameters in anomaly class for window id's
_C.PARAMS.img_win_id = 111
_C.PARAMS.rec_win_id = 112
_C.PARAMS.plt_win_id = 222


# -----------------------------------------------------------------------------
# TRAIN 
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
# Parameters for train_ae class 
_C.TRAIN.learning_rate = 0.001
_C.TRAIN.num_epochs = 50
_C.TRAIN.batch_size = 512

# -----------------------------------------------------------------------------
# DIR - # directory save paths
# -----------------------------------------------------------------------------
_C.DIR = CN()
_C.DIR.saved_models = "saved_models"
_C.DIR.base_dir = "<enter base directory here, branching to source, config etc."
_C.DIR.reconsLoss = "reconsLoss csv save path. Sample - reconsLoss.csv"
_C.DIR.testclip = "testclip csv save path. Sample - 'testclip1.csv'"

# -----------------------------------------------------------------------------
# TEST - # Test paths
# -----------------------------------------------------------------------------

_C.TEST = CN()
_C.TEST.gpu = True
_C.TEST.enc_dec_model = "epoch_30_pt"

_C.TEST.anomaly_data_path = "Path for test sample with anomaly class"
_C.TEST.data_path = "path for test sample with autoencoder class train_ae"
_C.TEST.save_path = "path for save with autoencoder class train_ae"
