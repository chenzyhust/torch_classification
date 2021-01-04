import os
from datetime import datetime
from yacs.config import CfgNode as CN

_C = CN()

#global settings
_C.CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343) 
_C.CIFAR100_TRAIN_STD  = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
_C.CHECKPOINT_PATH     = 'checkpoint' # directory to save weights file
_C.DATE_FORMAT         = '%A_%d_%B_%Y_%Hh_%Mm_%Ss' # time of we run the script
_C.TIME_NOW            = datetime.now().strftime(_C.DATE_FORMAT)
_C.LOG_DIR             = 'runs'      # tensorboard log dir
_C.SAVE_EPOCH          = 10          # save weights file per SAVE_EPOCH epoch
_C.GPU                 = True        # whether to use GPU
_C.NET                 = "resnet50"  # net type
_C.DIST                = True       # torch.distributed
_C.LOCAL_RANK          = 0
_C.DEVICE              = 'cuda'      # cuda or cpu
_C.SYNC_BN             = False       

# train parameters 
_C.TRAIN = CN()
_C.TRAIN.DATASET     = "cifar100"        # dataset type
_C.TRAIN.CLASSES     = 100               # num of classes
_C.TRAIN.EPOCHES     = 200               # total training epoches
_C.TRAIN.BATCH       = 128               # batch size dp : ALL=BATCH  dist : ALL=BATCH * world_size
_C.TRAIN.WARM        = 5                 # warm up epoches
_C.TRAIN.OPTIM       = "sgd"             # optim type 
_C.TRAIN.LR          = 0.4              # initial learning rate
_C.TRAIN.SCHEDULER   = "cosine"          # learning rate adjust type step or cosine
_C.TRAIN.STEPS       = [60, 120, 160]
_C.TRAIN.RESUME      = False             # resume training
_C.TRAIN.GPU_ID      = "0,1,2,3"         # id(s) for CUDA_VISIBLE_DEVICES
_C.TRAIN.CUDNN       = True              # cudnn.benchmark 
_C.TRAIN.SEED        = 0

# aug strategy
_C.AUG = CN()
_C.AUG.PROB         = 1           # probability of data aug
#                   : mixup
_C.AUG.MIXUP        = False       # whether to use mixup
_C.AUG.MIXUP_A      = 0.5         # mixup alpha paramter
#                   : cutout & dual cutout
_C.AUG.CUTOUT       = False       # whether to use cutout
_C.AUG.D_CUTOUT     = False       # whether to use dualcutout
_C.AUG.N_HOLES      = 1           # number of holes to cut out from image
_C.AUG.LENGTH       = 8           # length of the holes
_C.AUG.DCUT_ALPHA   = 0.1         # dual cutout loss parameter
#                   : random erasing
_C.AUG.R_ERASING    = False       # whether to use cutout
_C.AUG.M_ATTEMPT    = 20          # num of max attempt
_C.AUG.A_RAT_RANGE  = [0.02, 0.4] # area ratio range
_C.AUG.M_ASP_RATIO  = 0.3         # min aspect ration
#                   : cutmix
_C.AUG.CUTMIX       = False       # whether to use cutmix
_C.AUG.CUTMIX_BETA  = 1           # hyperparameter beta of cutmix
#                   : ricap
_C.AUG.RICAP        = False       # whether to use ricap
_C.AUG.RICAP_BETA   = 0.3         # hyperparameter beta of ricap
#                   : label_smoothing
_C.AUG.LABEL_SMOOTH = False       # whether to use label_smoothing
_C.AUG.SMOOTH_EPS   = 0.1         # label_smoothing epsilon
#                   : focal_loss
_C.AUG.FOCAL_LOSS   = False       # whether to use focal_loss
_C.AUG.GAMMA        = 2.0         # focal loss gamma
_C.AUG.ALPHA        = 0.25        # focal loss alpha

# test parameter
_C.TEST = CN()
_C.TEST.WEIGHT  = "./checkpoint/resnet50/Friday_01_January_2021_14h_45m_41s/resnet50-190-best.pth"
                                   # weight file path
_C.TEST.BATCH   = 128              # batch size
_C.TEST.GPU_ID  = "0,1,2,3"        # id(s) for CUDA_VISIBLE_DEVICES

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()
