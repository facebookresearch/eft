#conda activate venv_spin
from eft.apps.trainCaller import smplifyAllWrapper

params = ['--name','smplify_coco']

#Choose a pre-trained pose regressor to run EFT
# params +=['--pretrained_checkpoint','eft_model_zoo/mpii.pt','--num_workers','0','--noEval']     # A poor model
params +=['--pretrained_checkpoint','eft_model_zoo/cocoall_h36m_mpiinf_posetrack_lsptrain_ochuman.pt','--num_workers','0','--noEval']     # A strong model

params +=['--batch_size', '1']

#Choose DB
params +=['--db_set','coco']
params +=['--bExemplarMode']        #Required to turn off augmentation

# params +=['--bNotSkipExemplar']     #Always run, overwriting already processed one
params +=['--num_smplify_iters','100']    #50 would be sufficient in general
params +=['--bDebug_visEFT']            #To visualize EFT process. OpenGL requires screen, so cannot be used for servers


smplifyAllWrapper(params)    #original
