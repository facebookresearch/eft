# Original code from SPIN: https://github.com/nkolot/SPIN

"""
This file contains definitions of useful data stuctures and the paths
for the datasets and data files necessary to run the code.
Things you need to change: *_ROOT that indicate the path to each dataset
"""
import os
from os.path import join

bIsDevfair = False
if os.path.exists('/private/home/hjoo'):
    bIsDevfair= True

if bIsDevfair:
  ROOT_FOLDER ='/private/home/hjoo/codes/bodymocap/'

  H36M_ROOT = '/private/home/hjoo/data/h36m-fetch/human36m_10fps'
  H36M_TEST_ROOT = '/private/home/hjoo/data/h36m-fetch/extracted'

  LSP_ROOT = '/private/home/hjoo/data/lsp_dataset'
  LSP_ORIGINAL_ROOT = '/private/home/hjoo/data/lsp_dataset_original'
  LSPET_ROOT = '/private/home/hjoo/data/lspet_dataset/images_highres'
  MPII_ROOT = '/private/home/hjoo/data/mpii'
  COCO_ROOT = '/private/home/hjoo/data/coco'
  COCO2017_ROOT = '/private/home/hjoo/data/coco2017'
  COCO2017_SEMMAP_ROOT =''
  OCHUMAN_ROOT = '/private/home/hjoo/data/ochuman'
  MPI_INF_3DHP_ROOT = '/private/home/hjoo/data/mpi_inf_3dhp'
  PW3D_ROOT = '/private/home/hjoo/data/3dpw/'
  UPI_S1H_ROOT = ''

  #Our New Dataset
  POSETRACK_ROOT = '/private/home/hjoo/data/posetrack/images/train'
  PENNACTION_ROOT = '/private/home//hjoo/data/pennaction/frames'
  PANOPTICDB_ROOT = '/private/home//hjoo/data/panoptic_mtc/a4_release/hdImgs'
  # PANOPTICDB_SEQ_ROOT = '/private/home/hjoo/data/panoptic-toolbox' #General sequence
  PANOPTICDB_SEQ_ROOT = '/checkpoint/hjoo/panoptic-toolbox' #General sequence

  EXEMPLAR_OUTPUT_ROOT = '/private/home/hjoo/spinOut/'

  

else:
  ROOT_FOLDER ='./'   #/home/hjoo/codes/bodymocap/'
  H36M_ROOT = '/run/media/hjoo/disk/data/h36m-fetch/human36m_10fps_smplcocoTotalcap26_wShape_img'   #my folder format
  H36M_TEST_ROOT = '/run/media/hjoo/disk/data/h36m-fetch/extracted'         #spin folder format
  LSP_ROOT = '/run/media/hjoo/disk/data/lsp_dataset'
  LSP_ORIGINAL_ROOT = '/run/media/hjoo/disk/data/lsp_dataset_original'
  LSPET_ROOT = '/run/media/hjoo/disk/data/lspet_dataset/images_highres'
  MPII_ROOT = '/run/media/hjoo/disk/data/mpii_human_pose_v1'
  COCO_ROOT = '/run/media/hjoo/disk/data/coco'
  COCO2017_ROOT = '/run/media/hjoo/disk/data/coco2017'
  COCO2017_SEMMAP_ROOT = '/run/media/hjoo/disk/data/coco2017/annotations/panoptic_train2017_semmap'


  OCHUMAN_ROOT = '/run/media/hjoo/disk/data/OCHuman'
  MPI_INF_3DHP_ROOT = '/run/media/hjoo/disk/data/mpi_inf_3dhp'
  PW3D_ROOT = '/run/media/hjoo/disk/data/3dpw/'
  UPI_S1H_ROOT = ''

  POSETRACK_ROOT = '/run/media/hjoo/disk/data/posetrack/images'
  PENNACTION_ROOT = '/run/media/hjoo/disk/data/Penn_Action/frames'
  PANOPTICDB_ROOT = '/run/media/hjoo/disk/data/panoptic_mtc/a4_release/hdImgs'
  PANOPTICDB_SEQ_ROOT = '/home/hjoo/data/panoptic-toolbox' #General sequence

  EXEMPLAR_OUTPUT_ROOT = '/run/media/hjoo/disk/data/cvpr2020_eft_researchoutput/0_SPIN/0_exemplarOutputTest'

# Output folder to save test/train npz files
# DATASET_NPZ_PATH = ROOT_FOLDER+'data/dataset_extras'
DATASET_NPZ_PATH = ROOT_FOLDER+'preprocessdb/processeddata'

# Output folder to store the openpose detections
# This is requires only in case you want to regenerate 
# the .npz files with the annotations.
OPENPOSE_PATH = ROOT_FOLDER+'datasets/openpose'

# Path to test/train npz files
DATASET_FILES = [ {'h36m-p1': join(DATASET_NPZ_PATH, 'h36m_valid_protocol1.npz'),
                   'h36m-p2': join(DATASET_NPZ_PATH, 'h36m_valid_protocol2.npz'),
                   'lsp': join(DATASET_NPZ_PATH, 'lsp_dataset_test.npz'),
                   'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_valid.npz'),
                   '3dpw': join(DATASET_NPZ_PATH, '3dpw_test_withPt2d.npz'),
                   '3dpw-crop': join(DATASET_NPZ_PATH, '3dpw_test_uppercrop.npz'),#3dpw_test.npz'),
                  },

                  {

                  'h36m-p2': join(DATASET_NPZ_PATH, 'h36m_valid_protocol2.npz'),    #Fot testing time EFT
                  'h36m': join(DATASET_NPZ_PATH, 'h36m_training_fair_meter.npz'), #'h36m_single_train_openpose.npz'),
                   'lsp-orig': join(DATASET_NPZ_PATH, 'lsp_dataset_original_train.npz'),
                   'lsp-orig3d': join(DATASET_NPZ_PATH, 'lsp3d_11-14_lsp_analysis_100.npz'),
                   'mpii': join(DATASET_NPZ_PATH, 'mpii_train.npz'),
                   'coco': join(DATASET_NPZ_PATH, 'coco_2014_train_annotid.npz'), #coco SAMPLES with more thna 12 kps valid (used in SPIN)
                   'cocoall': join(DATASET_NPZ_PATH, 'coco_2014_train_6kp.npz'),    #coco SAMPLES with more than 6 kps valid
                   'cocofoot': join(DATASET_NPZ_PATH, 'coco_2014_train_12kp_foot.npz'),   #With GT Foot annotations. To run EFT
                   'cocofoot3d': join(DATASET_NPZ_PATH, '04-20_cocofoot_with8143_annotId.npz'),   #from EFT out with GT Foot. To train

                   'coco2017_whole_train_6kp': join(DATASET_NPZ_PATH, 'coco2017_wholebody_train_v10_6kp.npz'),   #COCO_whole with face body hand annotations
                   'coco2017_whole_train_12kp': join(DATASET_NPZ_PATH, 'coco2017_wholebody_train_v10_12kp.npz'),   #COCO_whole with face body hand annotations

                   'coco2014_train_6kp_semmap': join(DATASET_NPZ_PATH, 'coco_semmap3d_05-25_cocoall_with1336_iterThr.npz'),   #COCO 2014 with semmantic map name (2017 folder)

                   'ochuman': join(DATASET_NPZ_PATH, 'ochuman.npz'),
                   'ochuman3d': join(DATASET_NPZ_PATH, 'ochuman3d_01-28_ochuman_with8143_annotId.npz'),
                   'lspet': join(DATASET_NPZ_PATH, 'hr-lspet_train.npz'),
                   'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_train.npz'),

                   'coco3d': join(DATASET_NPZ_PATH, '04-14_coco_with8143_annotId.npz'),    #coco SAMPLES with more than 12 kps valid
                   'cocoall3d': join(DATASET_NPZ_PATH, '04-14_cocoall_with8143_annotId.npz'), #coco SAMPLES with more than 6 kps valid
                  #  'cocoall3d': join(DATASET_NPZ_PATH, 'coco3d_05-18_cocoall_with1336.npz '), #coco SAMPLES with more than 6 kps valid
                   'cocoplus3d': join(DATASET_NPZ_PATH, 'coco_2014_train_missingLimbs.npz'),#coco_2014_train_missingLimbs.npz'),    #outdated. more than 6kps less than 12 kps valid

                   'mpii3d': join(DATASET_NPZ_PATH, 'mpii3d_11-13_mpii_analysis_50_fter65.npz'),    #Should be written by trainOption
                   'lspet3d': join(DATASET_NPZ_PATH, 'lspet3d_11-08_lspet_with8143.npz'), #Should be written by trainOption
                   'posetrack3d' :join(DATASET_NPZ_PATH, ''),#10-09-posetrack-train_validlimbs.npz'),

                   'pennaction' :join(DATASET_NPZ_PATH, '10-09-exempler_pennaction_noFilter.npz'),#pennaction3d_11-04_pennaction_originalCode_weak_meta.npz'),#10-09-exempler_pennaction_validlimbs.npz'),
                  #  'panoptic' :join(DATASET_NPZ_PATH, 'panopticDB.npz')
                  #  'panoptic' :join(DATASET_NPZ_PATH, 'panoptic3d_11-05_panoptic_refit.npz')#,#panoptic3d_11-05_panoptic_initFit_21422.npz'), #panopticDB.npz')
                    # 'panoptic' :join(DATASET_NPZ_PATH, 'panopticDB.npz'), #panopticdb all views
                    'panoptichand' :join(DATASET_NPZ_PATH, 'panopticDB_extopview_hand.npz'), #panopticdb excluding top views (1,2,4,6,7,13,17,19)
                    'panoptic' :join(DATASET_NPZ_PATH, 'panopticDB_extopview.npz'), #panopticdb excluding top views (1,2,4,6,7,13,17,19)
                    'panoptic3d' :join(DATASET_NPZ_PATH, 'panoptic3d_04-23_panoptic_with8143_iter60_smpl.npz'), #panopticdb excluding top views (1,2,4,6,7,13,17,19)
                    'panoptic_haggling_test' :join(DATASET_NPZ_PATH, 'panoptic_haggling_testing.npz'), #any panoptic seq data for eft fitting

                   '3dpw_test' :join(DATASET_NPZ_PATH, '3dpw_test_withPt2d.npz'),#only for debuggin
                   '3dpw_test_crop': join(DATASET_NPZ_PATH, '3dpw_test_uppercrop.npz'),#3dpw_test.npz'),
                   '3dpw_train': join(DATASET_NPZ_PATH, '3dpw_train.npz'),#3dpw_test.npz'),
                  #  '3dpw_test_pt2d': join(DATASET_NPZ_PATH, '3dpw_test_withPt2d.npz')#3dpw_test.npz'),
                  }
                ]

DATASET_FOLDERS = {'h36m': H36M_ROOT,
                   'h36m-p1': H36M_TEST_ROOT,
                   'h36m-p2': H36M_TEST_ROOT,
                   'lsp-orig': LSP_ORIGINAL_ROOT,
                   'lsp-orig3d': LSP_ORIGINAL_ROOT,
                   'lsp': LSP_ROOT,
                   'lspet': LSPET_ROOT,
                   'mpi-inf-3dhp': MPI_INF_3DHP_ROOT,
                   'mpii': MPII_ROOT,
                   'coco': COCO_ROOT,
                   'cocofoot': COCO_ROOT,
                   'ochuman': OCHUMAN_ROOT,
                   'ochuman3d': OCHUMAN_ROOT,
                   'cocoall': COCO_ROOT,
                   'coco3d': COCO_ROOT,
                   'cocoall3d': COCO_ROOT,
                   'cocofoot3d': COCO_ROOT,

                   'coco2017_whole_train_6kp': COCO2017_ROOT,
                   'coco2017_whole_train_12kp': COCO2017_ROOT,
                   'coco2014_train_6kp_semmap': COCO2017_SEMMAP_ROOT,
                   

                   'mpii3d': MPII_ROOT,
                   'lspet3d': LSPET_ROOT,
                   'cocoplus3d': COCO_ROOT,
                   'posetrack3d': POSETRACK_ROOT,

                   '3dpw': PW3D_ROOT,
                   '3dpw-crop': PW3D_ROOT,
                   'upi-s1h': UPI_S1H_ROOT,
                   'pennaction': PENNACTION_ROOT,
                   'panoptic': PANOPTICDB_ROOT,
                   'panoptic3d': PANOPTICDB_ROOT,
                   'panoptichand': PANOPTICDB_ROOT,
                   'panoptic_haggling_test': PANOPTICDB_SEQ_ROOT,
                   '3dpw_test': PW3D_ROOT,
                   '3dpw_test_crop': PW3D_ROOT,
                   '3dpw_train': PW3D_ROOT,
                }


CUBE_PARTS_FILE = ROOT_FOLDER+'data/cube_parts.npy'
JOINT_REGRESSOR_TRAIN_EXTRA = ROOT_FOLDER+'extradata/data_from_spin//J_regressor_extra.npy'
JOINT_REGRESSOR_TRAIN_EXTRA_SMPLX = ROOT_FOLDER+'extradata/data_from_spin/J_regressor_extra_smplx.npy'    #For SMPLX
JOINT_REGRESSOR_H36M = ROOT_FOLDER+'data/J_regressor_h36m.npy'
VERTEX_TEXTURE_FILE = ROOT_FOLDER+'data/vertex_texture.npy'
STATIC_FITS_DIR = ROOT_FOLDER+'data/static_fits'
SMPL_MEAN_PARAMS = ROOT_FOLDER+'extradata/data_from_spin//smpl_mean_params.npz'
# SMPL_MODEL_DIR = ROOT_FOLDER+'smpl/'
SMPL_MODEL_DIR = ROOT_FOLDER+'extradata/smpl/'


def SetDBName(dbname, filename):
    is_train = 1
    DATASET_FILES[is_train][dbname] = join(DATASET_NPZ_PATH, filename)

    print(">>> set dbName of {}: {} ".format(dbname, DATASET_FILES[is_train][dbname]))

    