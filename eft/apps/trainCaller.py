import sys

from eft.utils.train_options import TrainOptions
from eft.train import EFTFitter
from eft.datasets import BaseDataset

def exemplarTrainerWrapper(params):
    
    print("Trainer function is called")
    options = TrainOptions().parse_args(params)
    # trainer = Trainer_weakpers(options)       #Not used anymore
    # trainer = Trainer(options) 
    eftFitter = EFTFitter(options) 
    eftFitter.eftAllInDB()


# def eft3DPWTestWrapper(params):
    
#     print("Trainer function is called")
#     options = TrainOptions().parse_args(params)
#     # trainer = Trainer_weakpers(options)       #Not used anymore
#     # trainer = Trainer(options) 
#     eftFitter = EFTFitter(options) 
#     # eftFitter.eftAllInDB()
#     eftFitter.eftAllInDB_3dpwtest()

def smplifyAllWrapper(params):
    
    print("Trainer function is called")
    options = TrainOptions().parse_args(params)
    # trainer = Trainer_weakpers(options)       #Not used anymore
    # trainer = Trainer(options) 
    eftFitter = EFTFitter(options) 
    eftFitter.smplifyAllInDB()


def exemplarTrainerWrapper_analysis(params):
    
    print("exemplarTrainerWrapper_analysisfunction is called")
    options = TrainOptions().parse_args(params)     
    # trainer = Trainer_weakpers(options)   #Not used anymore
    # trainer = Trainer(options)
    trainer = EFTFitter(options) 

    if options.bExemplar_badsample_finder:
        test_dataset_3dpw = BaseDataset(options, '3dpw', is_train=False,bMiniTest =True)
        test_dataset_h36m = BaseDataset(options, 'h36m-p1', is_train=False,bMiniTest =True)
    else:
        test_dataset_3dpw = BaseDataset(options, '3dpw', is_train=False,bMiniTest =True)
        test_dataset_h36m = BaseDataset(options, 'h36m-p1', is_train=False,bMiniTest =True)

    trainer.eftAllInDB(test_dataset_3dpw, test_dataset_h36m)


if __name__ == '__main__':
    if len(sys.argv)>1:
        options = TrainOptions().parse_args()
    else:
        # params =['--name','train_example','--pretrained_checkpoint','mymodels/2019_10_30-14_12_24.pt','--num_workers',0,'--db_set','ori_all','--batch_size','8']
        # # params =['--name','train_example','--pretrained_checkpoint','data/model_checkpoint.pt','--num_workers',0 ,'--db_set','mpii3d','--batch_size','8','--noEval']
        # params =['--name','train_example','--pretrained_checkpoint','logs/10-31-79034-wspin_all/checkpoints/2019_10_31-23_15_15.pt','--num_workers',0 ,'--db_set','mpii3d','--batch_size','128','--noEval']
        # params =['--name','train_example','--pretrained_checkpoint','/home/hjoo/Dropbox (Facebook)/spinouput/10-31-79034-wspin_all/checkpoints/2019_10_31-23_15_15.pt','--num_workers',0 ,'--db_set','coco3d','--batch_size','8','--noEval']

        params =['--name','test','--num_workers',0 ,'--db_set','ochuman','--batch_size','1','--noEval']#, '--run_smplify']
        # params +=['--pretrained_checkpoint','data/model_checkpoint.pt']
        params +=['--pretrained_checkpoint','/home/hjoo/Dropbox (Facebook)/spinouput/11-07-46582-w_upper0_2_ours_lc3d_all-8143/checkpoints/2019_11_07-17_32_54-best-55.422715842723846.pt']
        # param = ['--pretrained_checkpoint','/home/hjoo/Dropbox (Facebook)/spinouput/11-04-59953-filShp3_ours_coco3d_only-5198/checkpoints/2019_11_05-10_10_40.pt']
        # params +=['--db_mpii3d_name','mpii3d_11-01-52066_mpii_naiveBeta_noj3d.npz']
        params +=['--db_mpii3d_name','mpii3d_11-06_mpii_legOriLoss.npz']
        params +=['--db_coco3d_name','coco3d_11-06_coco_legOriLoss.npz']
        params +=['--db_cocoplus3d_name','coco3d_11-08_cocoplus_with8143.npz']
        params +=['--db_lspet3d_name','lspet3d_11-06_lspet_legOriLoss.npz']
        params +=['--bExemplarMode']
        # params +=['--bUseWeakProj']
        # params +=['--bUseKneePrior']
        # params +=['--bExemplarWith3DSkel']
        # params +=['--bExemplarWith3DSkel']
        params +=['--bNotSkipExemplar']     #Always run
        params +=['--maxExemplarIter','200']
        # params +=['--bUpperBodyTest', '--upperBodyTest_prob','1.0']
        # params +=['--bFacePartTest']

        params +=['--bDebug_visEFT']        
        params +=['--num_workers',0,'--noEval']
        # params +=['--bExemplar_analysis_testloss']
        # params +=['--bExemplar_dataLoaderStart','15939']
        # params +=['--run_smplify']

        options = TrainOptions().parse_args(params)
    
    if False:
        print("Run Trainer_weakpers!!")
        trainer = Trainer_weakpers(options)
    else:
        print("Run Original Trainer!!")
        trainer = Trainer(options)

    test_dataset_3dpw = BaseDataset(options, '3dpw', is_train=False,bMiniTest =True)
    test_dataset_h36m = BaseDataset(options, 'h36m-p1', is_train=False,bMiniTest =True)


    if options.bExemplarMode:
        trainer.eftAllInDB(test_dataset_3dpw, test_dataset_h36m)
    else:
        trainer.train(test_dataset_3dpw, test_dataset_h36m)
