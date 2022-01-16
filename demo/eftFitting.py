from eft.apps.trainCaller import exemplarTrainerWrapper

params = ['--name','eft_coco']

#Choose a pre-trained pose regressor to run EFT
# params +=['--pretrained_checkpoint','eft_model_zoo/mpii.pt','--num_workers','0','--noEval']     # A poor model
params +=['--pretrained_checkpoint','eft_model_zoo/cocoall_h36m_mpiinf_posetrack_lsptrain_ochuman.pt','--num_workers','0','--noEval']     # A strong model


params +=['--batch_size', '1']      #only one sample for EFT

#Choose DB
params +=['--db_set','coco']        #target DB to run EFT fitting. DB names are defined in ./eft/cores/config.py
params +=['--bExemplarMode']        #Turn on EFT mode option in training model

#For bad model
params +=['--eft_thresh_keyptErr_2d','2e-5']    
params +=['--maxExemplarIter','20']    #50 would be more than sufficient in general
params +=['--bDebug_visEFT']      #To visualize EFT process. OpenGL requires screen. Comment out this if you want to run it off-screen.

exemplarTrainerWrapper(params)    
