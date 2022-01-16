"""
This file contains the definition of different heterogeneous datasets used for training
"""
import torch
import numpy as np

from .base_dataset import BaseDataset
from os.path import join
import json
from eft.cores import config

class MixedDataset(torch.utils.data.Dataset):

    def __init__(self, options, **kwargs):
        # self.dataset_list = ['h36m', 'lsp-orig', 'mpii', 'lspet', 'coco', 'mpi-inf-3dhp']
        # self.dataset_dict = {'h36m': 0, 'lsp-orig': 1, 'mpii': 2, 'lspet': 3, 'coco': 4, 'mpi-inf-3dhp': 5}
        # self.dataset_list = ['lsp-orig', 'mpii', 'lspet', 'coco', 'mpi-inf-3dhp']
        # self.dataset_dict = {'lsp-orig': 1, 'mpii': 2, 'lspet': 3, 'coco': 4, 'mpi-inf-3dhp': 5}
        # self.dataset_list = ['lsp-orig', 'mpii', 'lspet', 'coco']
        # self.dataset_dict = {'lsp-orig': 1, 'mpii': 2, 'lspet': 3, 'coco': 4}

        # self.dataset_list = ['coco', 'mpi-inf-3dhp']
        # self.dataset_dict = {'coco': 4, 'mpi-inf-3dhp': 5}

        # self.dataset_list = ['mpi-inf-3dhp']
        # self.dataset_dict = {'mpi-inf-3dhp': 5}

        # self.dataset_list = [ 'mpii',  'coco']
        # self.dataset_dict = { 'mpii': 2, 'coco': 4}

        self.partition =[]

        length_itw =0

        if options.db_set == 'coco':
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'coco']
            self.dataset_dict = { 'coco': 0}
            self.partition = [1.0]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])
            
        elif options.db_set == 'coco-val':
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'coco-val']
            self.dataset_dict = { 'coco-val': 0}
            self.partition = [1.0]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])
        
        elif options.db_set == 'cocofoot':
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'cocofoot']
            self.dataset_dict = { 'cocofoot': 0}
            self.partition = [1.0]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])

        elif options.db_set == 'cocofoot3d':
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'cocofoot3d']
            self.dataset_dict = { 'cocofoot3d': 0}
            self.partition = [1.0]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])
        
        

        elif options.db_set == 'ochuman':
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'ochuman']
            self.dataset_dict = { 'ochuman': 0}
            self.partition = [1.0]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])

        elif options.db_set == 'ochuman3d':
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'ochuman3d']
            self.dataset_dict = { 'ochuman3d': 0}
            self.partition = [1.0]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])


        elif options.db_set == 'cocoall':
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'cocoall']
            self.dataset_dict = { 'cocoall': 0}
            self.partition = [1.0]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])

        elif options.db_set == 'cocoall3d':
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'cocoall3d']
            self.dataset_dict = { 'cocoall3d': 0}
            self.partition = [1.0]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])

        
        elif options.db_set == 'cocoplus3d':
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'cocoplus3d']
            self.dataset_dict = { 'cocoplus3d': 0}
            self.partition = [1.0]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])


        elif options.db_set == 'posetrack':
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'posetrack']
            self.dataset_dict = { 'posetrack': 0}
            self.partition = [1.0]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])

        elif options.db_set == 'posetrack3d':
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'posetrack3d']
            self.dataset_dict = { 'posetrack3d': 0}
            self.partition = [1.0]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])
        
        elif options.db_set == 'cocopart_posetrack3d':       
            self.dataset_list = ['coco3d', 'posetrack3d']
            self.dataset_dict = {'coco3d': 0, 'posetrack3d': 1}

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets])
            self.length = max([len(ds) for ds in self.datasets])

            self.partition = [len(self.datasets[0])/length_itw,
                                len(self.datasets[1])/length_itw]
            print("sampling rate: {}".format(self.partition))    
        
        elif options.db_set == 'cocoall_posetrack3d':       
            self.dataset_list = ['cocoall3d', 'posetrack3d']
            self.dataset_dict = {'cocoall3d': 0, 'posetrack3d': 1}

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets])
            self.length = max([len(ds) for ds in self.datasets])

            self.partition = [len(self.datasets[0])/length_itw,
                                 len(self.datasets[1])/length_itw]
            print("sampling rate: {}".format(self.partition))

        elif options.db_set == 'multicrop_cocoall_posetrack3d':       
            self.dataset_list = ['multicrop_cocoall3d', 'multicrop_posetrack3d']
            self.dataset_dict = {'multicrop_cocoall3d': 0, 'multicrop_posetrack3d': 1}

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets])
            self.length = max([len(ds) for ds in self.datasets])

            self.partition = [len(self.datasets[0])/length_itw,
                                 len(self.datasets[1])/length_itw]
            print("sampling rate: {}".format(self.partition))
        
        elif options.db_set == 'cocoall_posetrack3d_lspetatm':       
            self.dataset_list = ['cocoall3d', 'posetrack3d', 'lspet3d-amt']
            self.dataset_dict = {'cocoall3d': 0, 'posetrack3d': 1, 'lspet3d-amt':2}

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets])
            self.length = max([len(ds) for ds in self.datasets])

            self.partition = [len(self.datasets[0])/length_itw,
                                 len(self.datasets[1])/length_itw
                                 ,len(self.datasets[2])/length_itw]
            print("sampling rate: {}".format(self.partition))

        elif options.db_set == 'cocoall_posetrack3d_lspetatm-train':       
            self.dataset_list = ['cocoall3d', 'posetrack3d', 'lspet3d-amt-train']
            self.dataset_dict = {'cocoall3d': 0, 'posetrack3d': 1, 'lspet3d-amt-train':2}

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets])
            self.length = max([len(ds) for ds in self.datasets])

            self.partition = [len(self.datasets[0])/length_itw,
                                 len(self.datasets[1])/length_itw
                                 ,len(self.datasets[2])/length_itw]
            print("sampling rate: {}".format(self.partition))



        elif options.db_set == 'cocoall_mpii':       
            self.dataset_list = ['cocoall3d', 'mpii3d']
            self.dataset_dict = {'cocoall3d': 0, 'mpii3d': 1}

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets])
            self.length = max([len(ds) for ds in self.datasets])

            self.partition = [len(self.datasets[0])/length_itw,
                                 len(self.datasets[1])/length_itw]
            print("sampling rate: {}".format(self.partition))  
        

        elif options.db_set == 'cocoall_mpii_posetrack3d':       
            self.dataset_list = ['cocoall3d', 'mpii3d', 'posetrack3d']
            self.dataset_dict = {'cocoall3d': 0, 'mpii3d': 1, 'posetrack3d': 2}

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets])
            self.length = max([len(ds) for ds in self.datasets])

            self.partition = [len(self.datasets[0])/length_itw,
                                 len(self.datasets[1])/length_itw,
                                 len(self.datasets[2])/length_itw]
            print("sampling rate: {}".format(self.partition))  


        elif options.db_set == '3dpw_test':
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ '3dpw_test']
            self.dataset_dict = { '3dpw_test': 0}
            self.partition = [1.0]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])

        elif options.db_set == '3dpw_test_crop':
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ '3dpw_test_crop']
            self.dataset_dict = { '3dpw_test_crop': 0}
            self.partition = [1.0]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])
        
        elif options.db_set == '3dpw_test_multilevel_0':
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ '3dpw_test_multilevel_0']
            self.dataset_dict = { '3dpw_test_multilevel_0': 0}
            self.partition = [1.0]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])

        elif options.db_set == '3dpw_test_multilevel_1':
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ '3dpw_test_multilevel_1']
            self.dataset_dict = { '3dpw_test_multilevel_1': 0}
            self.partition = [1.0]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])
        
        elif options.db_set == '3dpw_test_multilevel_4':
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ '3dpw_test_multilevel_1']
            self.dataset_dict = { '3dpw_test_multilevel_1': 0}
            self.partition = [1.0]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])

        elif options.db_set == '3dpw_test_multilevel_7':
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ '3dpw_test_multilevel_7']
            self.dataset_dict = { '3dpw_test_multilevel_7': 0}
            self.partition = [1.0]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])
        
        elif options.db_set == '3dpw_train':
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ '3dpw_train']
            self.dataset_dict = { '3dpw_train': 0}
            self.partition = [1.0]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])

        elif options.db_set == 'h36m_p2_test':
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'h36m-p2']
            self.dataset_dict = { 'h36m-p2': 0}
            self.partition = [1.0]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])
            
        elif options.db_set == 'h36m_p2_test':
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'h36m-p2']
            self.dataset_dict = { 'h36m-p2': 0}
            self.partition = [1.0]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])
        

        elif options.db_set == 'pennaction':
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'pennaction']
            self.dataset_dict = { 'pennaction': 0}
            self.partition = [1.0]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])


        elif options.db_set == 'pennaction_all':
            self.dataset_list = ['h36m', 'mpi-inf-3dhp', 'lsp-orig', 'mpii', 'lspet', 'coco', 'pennaction' ]
            self.dataset_dict = {'h36m': 0, 'mpi-inf-3dhp': 1,  'lsp-orig': 2, 'mpii': 3, 'lspet': 4, 'coco': 5, 'pennaction': 6}

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[2:]])
            self.length = max([len(ds) for ds in self.datasets])

            self.partition = [.3, 0.1,
                          .6*len(self.datasets[2])/length_itw,
                          .6*len(self.datasets[3])/length_itw,
                          .6*len(self.datasets[4])/length_itw, 
                          .6*len(self.datasets[5])/length_itw,
                          .6*len(self.datasets[6])/length_itw]

            print("sampling rate: {}".format(self.partition))    

        elif options.db_set == 'panoptic':
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'panoptic']
            self.dataset_dict = { 'panoptic': 0}
            self.partition = [1.0]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])

        elif options.db_set == 'panoptic3d':

            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'panoptic3d']
            self.dataset_dict = { 'panoptic3d': 0}
            self.partition = [1.0]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])

        elif options.db_set == 'coco3d_panoptic3d':       #original all including h36m

            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'coco3d', 'panoptic3d']
            self.dataset_dict = { 'coco3d': 0, 'panoptic3d': 1}
            self.partition = [0.4, 0.6]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])
        
        elif options.db_set == 'cocoall3d_panoptic3d':       #original all including h36m

            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'cocoall3d', 'panoptic3d']
            self.dataset_dict = { 'cocoall3d': 0, 'panoptic3d': 1}
            self.partition = [0.4, 0.6]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])
        
        elif options.db_set == 'cocoall3d_pan3d_h36m':       #original all including h36m

            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'cocoall3d', 'panoptic3d', 'h36m']
            self.dataset_dict = { 'cocoall3d': 0, 'panoptic3d': 1, 'h36m': 2}
            self.partition = [0.4, 0.2, 0.4]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])
        
        elif options.db_set == 'cocoall3d_pan3d_h36m_ochuman3d':       #original all including h36m

            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'panoptic3d', 'h36m','cocoall3d', 'ochuman3d']
            self.dataset_dict = { 'panoptic3d': 0, 'h36m': 1, 'cocoall3d': 2,  'ochuman3d': 3}
            self.partition = [0.4, 0.1, 0.4, 0.1]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[2:]])
            self.length = max([len(ds) for ds in self.datasets])

            # self.partition = [.1, 0.3,
            #               .6*len(self.datasets[2])/length_itw,
            #               .6*len(self.datasets[3])/length_itw]

            self.partition = [.1, 0.3, 0.4, 0.2]

            print("sampling rate: {}".format(self.partition))   
            
        
        elif options.db_set == 'coco3d_pan3d_h36m':       #original all including h36m

            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'coco3d', 'panoptic3d', 'h36m']
            self.dataset_dict = { 'coco3d': 0, 'panoptic3d': 1, 'h36m': 2}
            self.partition = [0.4, 0.2, 0.4]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])


        elif options.db_set == 'panoptichand':
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'panoptichand']
            self.dataset_dict = { 'panoptichand': 0}
            self.partition = [1.0]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])

        elif options.db_set == 'panoptic_haggling_test':
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'panoptic_haggling_test']
            self.dataset_dict = { 'panoptic_haggling_test': 0}
            self.partition = [1.0]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])


        elif options.db_set == 'panoptic_all':
            self.dataset_list = ['h36m', 'mpi-inf-3dhp', 'panoptic', 'lsp-orig', 'mpii', 'lspet', 'coco' ]
            self.dataset_dict = {'h36m': 0, 'mpi-inf-3dhp': 1, 'panoptic':2, 'lsp-orig': 3, 'mpii': 4, 'lspet': 5, 'coco': 6}

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[3:]])
            self.length = max([len(ds) for ds in self.datasets])

            self.partition = [.2, 0.1, 0.1,
                          .6*len(self.datasets[3])/length_itw,
                          .6*len(self.datasets[4])/length_itw,
                          .6*len(self.datasets[5])/length_itw, 
                          .6*len(self.datasets[6])/length_itw]

            print("sampling rate: {}".format(self.partition))   

       
        
        elif options.db_set == 'h36m':
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'h36m']
            self.dataset_dict = { 'h36m': 0}
            self.partition = [1.0]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])

        elif options.db_set == 'lsp':
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'lsp-orig']
            self.dataset_dict = { 'lsp-orig': 0}
            self.partition = [1.0]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])

        elif options.db_set == 'lspet':
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'lspet']
            self.dataset_dict = { 'lspet': 0}
            self.partition = [1.0]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])

        elif options.db_set == 'lspet3d-amt':
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'lspet3d-amt']
            self.dataset_dict = { 'lspet3d-amt': 0}
            self.partition = [1.0]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])

        elif options.db_set == 'lspet3d-amt-train':
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'lspet3d-amt-train']
            self.dataset_dict = { 'lspet3d-amt-train': 0}
            self.partition = [1.0]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])




        elif options.db_set == 'minf':
            self.dataset_list = [ 'mpi-inf-3dhp']
            self.dataset_dict = { 'mpi-inf-3dhp': 0}
            self.partition = [1.0]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])

        elif options.db_set == 'mpii':
            self.dataset_list = [ 'mpii']
            self.dataset_dict = { 'mpii': 0}
            self.partition = [1.0]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])



        elif options.db_set == 'ori_all_noinf':

            self.dataset_list = ['h36m', 'lsp-orig', 'mpii', 'lspet', 'coco']
            self.dataset_dict = {'h36m': 0, 'lsp-orig': 1, 'mpii': 2, 'lspet': 3, 'coco': 4}

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])

            self.partition = [.35, .65*len(self.datasets[1])/length_itw,
                          .65*len(self.datasets[2])/length_itw,
                          .65*len(self.datasets[3])/length_itw, 
                          .65*len(self.datasets[4])/length_itw]
            print("sampling rate: {}".format(self.partition))    


        elif options.db_set == 'ori_all_noh36m':       #original all
            self.dataset_list = ['lsp-orig', 'mpii', 'lspet', 'coco', 'mpi-inf-3dhp']
            self.dataset_dict = {'lsp-orig': 1, 'mpii': 2, 'lspet': 3, 'coco': 4, 'mpi-inf-3dhp': 5}

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])

            self.partition = [.8*len(self.datasets[0])/length_itw,
                            .8*len(self.datasets[1])/length_itw,
                            .8*len(self.datasets[2])/length_itw, 
                            .8*len(self.datasets[3])/length_itw, 
                            0.2]
            print("sampling rate: {}".format(self.partition)) 


        elif options.db_set == 'ori_coco_h36m_inf':

            self.dataset_list = ['h36m','coco', 'mpi-inf-3dhp']
            self.dataset_dict = {'h36m': 0,'coco': 1, 'mpi-inf-3dhp': 2}

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:2]])
            self.length = max([len(ds) for ds in self.datasets])

            self.partition = [.3, .6*len(self.datasets[1])/length_itw,
                                0.1]
            print("sampling rate: {}".format(self.partition))  


        elif options.db_set == 'ori_all':       #original all including h36m
            self.dataset_list = ['h36m', 'lsp-orig', 'mpii', 'lspet', 'coco', 'mpi-inf-3dhp']
            self.dataset_dict = {'h36m': 0, 'lsp-orig': 1, 'mpii': 2, 'lspet': 3, 'coco': 4, 'mpi-inf-3dhp': 5}

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])

            self.partition = [.3, .6*len(self.datasets[1])/length_itw,
                          .6*len(self.datasets[2])/length_itw,
                          .6*len(self.datasets[3])/length_itw, 
                          .6*len(self.datasets[4])/length_itw,
                          0.1]
            print("sampling rate: {}".format(self.partition))    

        
        elif options.db_set == 'coco3d':       #original all including h36m
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'coco3d']
            self.dataset_dict = { 'coco3d': 0}
            self.partition = [1.0]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            # length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])

        elif options.db_set == 'coco3d_amt':       #original all including h36m
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'coco3d_amt']
            self.dataset_dict = { 'coco3d_amt': 0}
            self.partition = [1.0]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])

        
        #This dataset has multiple crop information (8 levels, 0 for head 8 for whole)
        elif options.db_set == 'multicrop_coco3dpart':       
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'multicrop_coco3d']
            self.dataset_dict = { 'multicrop_coco3d': 0}
            self.partition = [1.0]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])

        #This dataset has multiple crop information (8 levels, 0 for head 8 for whole)
        elif options.db_set == 'multicrop_cocoall3d':       
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'multicrop_cocoall3d']
            self.dataset_dict = { 'multicrop_cocoall3d': 0}
            self.partition = [1.0]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])


        elif options.db_set == 'coco2014_train_6kp_semmap':       #original all including h36m
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'coco2014_train_6kp_semmap']
            self.dataset_dict = { 'coco2014_train_6kp_semmap': 0}
            self.partition = [1.0]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])



        elif options.db_set == 'coco2017_whole_train_6kp':       #original all including h36m
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'coco2017_whole_train_6kp']
            self.dataset_dict = { 'coco2017_whole_train_6kp': 0}
            self.partition = [1.0]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])

        elif options.db_set == 'coco2017_whole_train_12kp':       #original all including h36m
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'coco2017_whole_train_12kp']
            self.dataset_dict = { 'coco2017_whole_train_12kp': 0}
            self.partition = [1.0]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])
        
        elif options.db_set == 'coco3d_cocoplus3d':       #original all including h36m

            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'coco3d', 'cocoplus3d']
            self.dataset_dict = { 'coco3d': 0, 'cocoplus3d': 1}
            
            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets])
            self.length = max([len(ds) for ds in self.datasets])

            self.partition = [len(self.datasets[0])/length_itw,
                            len(self.datasets[1])/length_itw]
            print("sampling rate: {}".format(self.partition))    

        elif options.db_set == 'coco3d_h36m':       #original all including h36m

            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'coco3d', 'h36m']
            self.dataset_dict = { 'coco3d': 0, 'h36m': 1}
            self.partition = [0.4, 0.6]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])


        #Default
        elif options.db_set == 'cocoall3d_h36m':
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'cocoall3d', 'h36m']
            self.dataset_dict = { 'cocoall3d': 0, 'h36m': 1}
            self.partition = [0.4, 0.6]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])

        elif options.db_set == 'cocoall3d_h36m_06':
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'cocoall3d', 'h36m']
            self.dataset_dict = { 'cocoall3d': 0, 'h36m': 1}
            self.partition = [0.4, 0.6]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])

        elif options.db_set == 'cocoall3d_h36m_05':
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'cocoall3d', 'h36m']
            self.dataset_dict = { 'cocoall3d': 0, 'h36m': 1}
            self.partition = [0.5, 0.5]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])
        
        elif options.db_set == 'cocoall3d_h36m_04':
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'cocoall3d', 'h36m']
            self.dataset_dict = { 'cocoall3d': 0, 'h36m': 1}
            self.partition = [0.6, 0.4]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])
        
        elif options.db_set == 'cocoall3d_h36m_03':
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'cocoall3d', 'h36m']
            self.dataset_dict = { 'cocoall3d': 0, 'h36m': 1}
            self.partition = [0.7, 0.3]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])
        
        elif options.db_set == 'cocoall3d_h36m_02':
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'cocoall3d', 'h36m']
            self.dataset_dict = { 'cocoall3d': 0, 'h36m': 1}
            self.partition = [0.8, 0.2]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])
        
        elif options.db_set == 'cocoall3d_h36m_01':
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'cocoall3d', 'h36m']
            self.dataset_dict = { 'cocoall3d': 0, 'h36m': 1}
            self.partition = [0.9, 0.1]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])

        elif options.db_set == 'coco3d_all':       #original all including h36m
            self.dataset_list = ['h36m', 'lsp-orig', 'mpii', 'lspet', 'coco3d', 'mpi-inf-3dhp']
            self.dataset_dict = {'h36m': 0, 'lsp-orig': 1, 'mpii': 2, 'lspet': 3, 'coco3d': 4, 'mpi-inf-3dhp': 5}

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])

            self.partition = [.3, .6*len(self.datasets[1])/length_itw,
                          .6*len(self.datasets[2])/length_itw,
                          .6*len(self.datasets[3])/length_itw, 
                          .6*len(self.datasets[4])/length_itw,
                          0.1]
            print("sampling rate: {}".format(self.partition))    

        elif options.db_set == 'mpii3d':       
            self.dataset_list = ['mpii3d']
            self.dataset_dict = {'mpii3d': 0}

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])

            self.partition = [1.0]

        elif options.db_set == 'mpii3d_all':   

            self.dataset_list = ['h36m', 'lsp-orig', 'mpii3d', 'lspet', 'coco', 'mpi-inf-3dhp']
            self.dataset_dict = {'h36m': 0, 'lsp-orig': 1, 'mpii3d': 2, 'lspet': 3, 'coco': 4, 'mpi-inf-3dhp': 5}

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])

            self.partition = [.3, .6*len(self.datasets[1])/length_itw,
                          .6*len(self.datasets[2])/length_itw,
                          .6*len(self.datasets[3])/length_itw, 
                          .6*len(self.datasets[4])/length_itw,
                          0.1]
            print("sampling rate: {}".format(self.partition)) 

            


        elif options.db_set == 'coco3d_mpii3d_all':       
            self.dataset_list = ['h36m', 'lsp-orig', 'mpii3d', 'lspet', 'coco3d', 'mpi-inf-3dhp']
            self.dataset_dict = {'h36m': 0, 'lsp-orig': 1, 'mpii3d': 2, 'lspet': 3, 'coco3d': 4, 'mpi-inf-3dhp': 5}

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])

            self.partition = [.3, .6*len(self.datasets[1])/length_itw,
                          .6*len(self.datasets[2])/length_itw,
                          .6*len(self.datasets[3])/length_itw, 
                          .6*len(self.datasets[4])/length_itw,
                          0.1]
            print("sampling rate: {}".format(self.partition))    

        elif options.db_set == 'lspet3d':       
            self.dataset_list = ['lspet3d']
            self.dataset_dict = {'lspet3d': 0}

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])

            self.partition = [1.0]


        elif options.db_set == 'lspet3d_all':   

            self.dataset_list = ['h36m', 'lsp-orig', 'mpii', 'lspet3d', 'coco', 'mpi-inf-3dhp']
            self.dataset_dict = {'h36m': 0, 'lsp-orig': 1, 'mpii': 2, 'lspet3d': 3, 'coco': 4, 'mpi-inf-3dhp': 5}

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])

            self.partition = [.3, .6*len(self.datasets[1])/length_itw,
                          .6*len(self.datasets[2])/length_itw,
                          .6*len(self.datasets[3])/length_itw, 
                          .6*len(self.datasets[4])/length_itw,
                          0.1]
            print("sampling rate: {}".format(self.partition)) 

        elif options.db_set == 'mpii3d_lspet3d_all':       
            self.dataset_list = ['h36m', 'lsp-orig', 'mpii3d', 'lspet3d', 'coco', 'mpi-inf-3dhp']
            self.dataset_dict = {'h36m': 0, 'lsp-orig': 1, 'mpii3d': 2, 'lspet3d': 3, 'coco': 4, 'mpi-inf-3dhp': 5}

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])

            self.partition = [.3, .6*len(self.datasets[1])/length_itw,
                          .6*len(self.datasets[2])/length_itw,
                          .6*len(self.datasets[3])/length_itw, 
                          .6*len(self.datasets[4])/length_itw,
                          0.1]
            print("sampling rate: {}".format(self.partition))    

        elif options.db_set == 'mlc3d_all':       
            self.dataset_list = ['h36m', 'lsp-orig', 'mpii3d', 'lspet3d', 'coco3d', 'mpi-inf-3dhp']
            self.dataset_dict = {'h36m': 0, 'lsp-orig': 1, 'mpii3d': 2, 'lspet3d': 3, 'coco3d': 4, 'mpi-inf-3dhp': 5}

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])

            self.partition = [.3, .6*len(self.datasets[1])/length_itw,
                          .6*len(self.datasets[2])/length_itw,
                          .6*len(self.datasets[3])/length_itw, 
                          .6*len(self.datasets[4])/length_itw,
                          0.1]
            print("sampling rate: {}".format(self.partition))    

        
        elif options.db_set == 'mllc3d_all':       
            self.dataset_list = ['h36m', 'lsp-orig3d', 'mpii3d', 'lspet3d', 'coco3d', 'mpi-inf-3dhp']
            self.dataset_dict = {'h36m': 0, 'lsp-orig3d': 1, 'mpii3d': 2, 'lspet3d': 3, 'coco3d': 4, 'mpi-inf-3dhp': 5}

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])

            self.partition = [.3, .6*len(self.datasets[1])/length_itw,
                          .6*len(self.datasets[2])/length_itw,
                          .6*len(self.datasets[3])/length_itw, 
                          .6*len(self.datasets[4])/length_itw,
                          0.1]
            print("sampling rate: {}".format(self.partition))    
        

        ################################################################################################################
        elif options.db_set == 'eft_spindb_all':       
            self.dataset_list = ['h36m', 'mpii3d', 'lspet3d', 'coco3d', 'mpi-inf-3dhp']
            self.dataset_dict = {'h36m': 0, 'mpii3d': 1, 'lspet3d': 2, 'coco3d': 3, 'mpi-inf-3dhp': 4}

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])

            self.partition = [.3,
                          .6*len(self.datasets[1])/length_itw,
                          .6*len(self.datasets[2])/length_itw, 
                          .6*len(self.datasets[3])/length_itw,
                          0.1]
            print("\nsampling rate: {}".format(self.partition))    

        elif options.db_set == 'eft_spindb_all_pan3d':       
            self.dataset_list = ['h36m', 'mpii3d', 'lspet3d', 'coco3d', 'mpi-inf-3dhp', 'panoptic3d']
            self.dataset_dict = {'h36m': 0, 'mpii3d': 1, 'lspet3d': 2, 'coco3d': 3, 'mpi-inf-3dhp': 4, 'panoptic3d': 5}

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-2]])
            self.length = max([len(ds) for ds in self.datasets])

            self.partition = [.3,
                          .55*len(self.datasets[1])/length_itw,
                          .55*len(self.datasets[2])/length_itw,
                          .55*len(self.datasets[3])/length_itw, 
                          0.1, 0.05]
            print("sampling rate: {}".format(self.partition))    
        
        
        elif options.db_set == 'eft_spindb_all_cocoall':       
            self.dataset_list = ['h36m', 'mpii3d', 'lspet3d', 'cocoall3d', 'mpi-inf-3dhp']
            self.dataset_dict = {'h36m': 0, 'mpii3d': 1, 'lspet3d': 2, 'cocoall3d': 3, 'mpi-inf-3dhp': 4}

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])

            self.partition = [.3,
                          .6*len(self.datasets[1])/length_itw,
                          .6*len(self.datasets[2])/length_itw, 
                          .6*len(self.datasets[3])/length_itw,
                          0.1]
            print("sampling rate: {}".format(self.partition))    

        elif options.db_set == 'eft_spindb_all_pan3d_cocoall':       
            self.dataset_list = ['h36m', 'mpii3d', 'lspet3d', 'cocoall3d', 'mpi-inf-3dhp', 'panoptic3d']
            self.dataset_dict = {'h36m': 0, 'mpii3d': 1, 'lspet3d': 2, 'cocoall3d': 3, 'mpi-inf-3dhp': 4, 'panoptic3d': 5}

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-2]])
            self.length = max([len(ds) for ds in self.datasets])

            self.partition = [.3,
                          .55*len(self.datasets[1])/length_itw,
                          .55*len(self.datasets[2])/length_itw,
                          .55*len(self.datasets[3])/length_itw, 
                          0.1, 0.05]
            print("sampling rate: {}".format(self.partition))    
        
        ################################################################################################################
        



        elif options.db_set == 'panoptic_ours3d_all':
            self.dataset_list = ['h36m', 'mpi-inf-3dhp', 'panoptic3d', 'lsp-orig', 'mpii3d', 'lspet3d', 'coco3d' ]
            self.dataset_dict = {'h36m': 0, 'mpi-inf-3dhp': 1, 'panoptic3d':2, 'lsp-orig': 3, 'mpii3d': 4, 'lspet3d': 5, 'coco3d': 6}

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[3:]])
            self.length = max([len(ds) for ds in self.datasets])

            self.partition = [.3, 0.1, 0.1,
                          .5*len(self.datasets[3])/length_itw,
                          .5*len(self.datasets[4])/length_itw,
                          .5*len(self.datasets[5])/length_itw, 
                          .5*len(self.datasets[6])/length_itw]

            print("sampling rate: {}".format(self.partition))    



        elif options.db_set == 'mc3d_all':       
            self.dataset_list = ['h36m', 'lsp-orig', 'mpii3d', 'lspet', 'coco3d', 'mpi-inf-3dhp']
            self.dataset_dict = {'h36m': 0, 'lsp-orig': 1, 'mpii3d': 2, 'lspet': 3, 'coco3d': 4, 'mpi-inf-3dhp': 5}

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])

            self.partition = [.3, .6*len(self.datasets[1])/length_itw,
                          .6*len(self.datasets[2])/length_itw,
                          .6*len(self.datasets[3])/length_itw, 
                          .6*len(self.datasets[4])/length_itw,
                          0.1]
            print("sampling rate: {}".format(self.partition))    

        elif options.db_set == 'lc3d_all':       
            self.dataset_list = ['h36m', 'lsp-orig', 'mpii', 'lspet3d', 'coco3d', 'mpi-inf-3dhp']
            self.dataset_dict = {'h36m': 0, 'lsp-orig': 1, 'mpii': 2, 'lspet3d': 3, 'coco3d': 4, 'mpi-inf-3dhp': 5}

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])

            self.partition = [.3, .6*len(self.datasets[1])/length_itw,
                          .6*len(self.datasets[2])/length_itw,
                          .6*len(self.datasets[3])/length_itw, 
                          .6*len(self.datasets[4])/length_itw,
                          0.1]
            print("sampling rate: {}".format(self.partition))    


        elif options.db_set == 'lc3d_all_cocoplus':
            self.dataset_list = ['h36m', 'mpi-inf-3dhp', 'lspet3d', 'coco3d', 'cocoplus3d' ]
            self.dataset_dict = {'h36m': 0, 'mpi-inf-3dhp': 1, 'lspet3d': 2, 'coco3d': 3, 'cocoplus3d': 4}

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[2:]])
            self.length = max([len(ds) for ds in self.datasets])

            self.partition = [.3, 0.1,
                          .6*len(self.datasets[2])/length_itw,
                          .6*len(self.datasets[3])/length_itw,
                          .6*len(self.datasets[4])/length_itw, 
                          ]
            print("sampling rate: {}".format(self.partition)) 
        
        elif options.db_set == 'lc3d_all_posetrack':  
            self.dataset_list = ['h36m', 'mpi-inf-3dhp', 'lspet3d', 'coco3d', 'posetrack3d' ]
            self.dataset_dict = {'h36m': 0, 'mpi-inf-3dhp': 1, 'lspet3d': 2, 'coco3d': 3, 'posetrack3d': 4}

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[2:]])
            self.length = max([len(ds) for ds in self.datasets])

            self.partition = [.3, 0.1,
                          .6*len(self.datasets[2])/length_itw,
                          .6*len(self.datasets[3])/length_itw,
                          .6*len(self.datasets[4])/length_itw, 
                          ]
            print("sampling rate: {}".format(self.partition)) 

        elif options.db_set == 'lc3d_all_cp_pt':  
            self.dataset_list = ['h36m', 'mpi-inf-3dhp', 'lspet3d', 'coco3d', 'cocoplus3d', 'posetrack3d' ]
            self.dataset_dict = {'h36m': 0, 'mpi-inf-3dhp': 1, 'lspet3d': 2, 'coco3d': 3, 'cocoplus3d': 4, 'posetrack3d': 5}

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[2:]])
            self.length = max([len(ds) for ds in self.datasets])

            self.partition = [.3, 0.1,
                          .6*len(self.datasets[2])/length_itw,
                          .6*len(self.datasets[3])/length_itw,
                          .6*len(self.datasets[4])/length_itw, 
                          .6*len(self.datasets[5])/length_itw,
                          ]
            print("sampling rate: {}".format(self.partition))   


        elif options.db_set == 'mlc3d_all_cocoplus':  
            self.dataset_list = ['h36m', 'mpi-inf-3dhp', 'mpii3d', 'lspet3d', 'coco3d', 'cocoplus3d' ]
            self.dataset_dict = {'h36m': 0, 'mpi-inf-3dhp': 1, 'mpii3d': 2, 'lspet3d': 3, 'coco3d': 4, 'cocoplus3d': 5}

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[2:]])
            self.length = max([len(ds) for ds in self.datasets])

            self.partition = [.3, 0.1,
                          .6*len(self.datasets[2])/length_itw,
                          .6*len(self.datasets[3])/length_itw,
                          .6*len(self.datasets[4])/length_itw, 
                          .6*len(self.datasets[5])/length_itw
                          ]
            print("sampling rate: {}".format(self.partition)) 

        elif options.db_set == 'mlc3d_all_posetrack':  
            self.dataset_list = ['h36m', 'mpi-inf-3dhp', 'mpii3d', 'lspet3d', 'coco3d', 'posetrack3d' ]
            self.dataset_dict = {'h36m': 0, 'mpi-inf-3dhp': 1, 'mpii3d': 2, 'lspet3d': 3, 'coco3d': 4, 'posetrack3d': 5}

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[2:]])
            self.length = max([len(ds) for ds in self.datasets])

            self.partition = [.3, 0.1,
                          .6*len(self.datasets[2])/length_itw,
                          .6*len(self.datasets[3])/length_itw,
                          .6*len(self.datasets[4])/length_itw, 
                          .6*len(self.datasets[5])/length_itw
                          ]
            print("sampling rate: {}".format(self.partition))    


      
        elif options.db_set == 'panoptic_ours3d_all_test':
            self.dataset_list = ['h36m', 'mpi-inf-3dhp', 'panoptic', 'lsp-orig', 'mpii3d', 'lspet3d', 'coco3d' ]
            self.dataset_dict = {'h36m': 0, 'mpi-inf-3dhp': 1, 'panoptic':2, 'lsp-orig': 3, 'mpii3d': 4, 'lspet3d': 5, 'coco3d': 6}

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[3:]])
            self.length = max([len(ds) for ds in self.datasets])

            self.partition = [.2, 0.0, 0.3,
                          .5*len(self.datasets[3])/length_itw,
                          .5*len(self.datasets[4])/length_itw,
                          .5*len(self.datasets[5])/length_itw, 
                          .5*len(self.datasets[6])/length_itw]

            print("sampling rate: {}".format(self.partition))    


        elif options.db_set == 'panoptic_ours3d_all_test2':
            self.dataset_list = ['h36m', 'mpi-inf-3dhp', 'panoptic', 'lsp-orig', 'mpii3d', 'lspet3d', 'coco3d' ]
            self.dataset_dict = {'h36m': 0, 'mpi-inf-3dhp': 1, 'panoptic':2, 'lsp-orig': 3, 'mpii3d': 4, 'lspet3d': 5, 'coco3d': 6}

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[3:]])
            self.length = max([len(ds) for ds in self.datasets])

            self.partition = [.1, 0.0, 0.4,
                          .5*len(self.datasets[3])/length_itw,
                          .5*len(self.datasets[4])/length_itw,
                          .5*len(self.datasets[5])/length_itw, 
                          .5*len(self.datasets[6])/length_itw]

            print("sampling rate: {}".format(self.partition))    

        elif options.db_set == 'mlc3d_all_cp_ps':  
            self.dataset_list = ['h36m', 'mpi-inf-3dhp', 'mpii3d', 'lspet3d', 'coco3d', 'cocoplus3d', 'posetrack3d' ]
            self.dataset_dict = {'h36m': 0, 'mpi-inf-3dhp': 1, 'mpii3d': 2, 'lspet3d': 3, 'coco3d': 4, 'cocoplus3d': 5, 'posetrack3d': 6}

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[2:]])
            self.length = max([len(ds) for ds in self.datasets])

            self.partition = [.3, 0.1,
                          .6*len(self.datasets[2])/length_itw,
                          .6*len(self.datasets[3])/length_itw,
                          .6*len(self.datasets[4])/length_itw, 
                          .6*len(self.datasets[5])/length_itw,
                          .6*len(self.datasets[6])/length_itw
                          ]
            print("sampling rate: {}".format(self.partition))    


        elif options.db_set == 'coco3d_plus3d_mpii3d':       
            self.dataset_list = ['coco3d','cocoplus3d', 'mpii3d']
            self.dataset_dict = {'coco3d': 0, 'cocoplus3d': 1, 'mpii3d': 2}

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets])
            self.length = max([len(ds) for ds in self.datasets])

            self.partition = [len(self.datasets[0])/length_itw,
                          len(self.datasets[1])/length_itw, 
                          len(self.datasets[2])/length_itw]
            print("sampling rate: {}".format(self.partition))    
        
        elif options.db_set == 'coco3d_plus3d_mpii3d_pt3d':       
            self.dataset_list = ['coco3d','cocoplus3d', 'mpii3d', 'posetrack3d']
            self.dataset_dict = {'coco3d': 0, 'cocoplus3d': 1, 'mpii3d': 2, 'posetrack3d': 3}

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets])
            self.length = max([len(ds) for ds in self.datasets])

            self.partition = [len(self.datasets[0])/length_itw,
                          len(self.datasets[1])/length_itw, 
                          len(self.datasets[2])/length_itw,
                          len(self.datasets[3])/length_itw,]
            print("sampling rate: {}".format(self.partition))    


        ################ w/ MPI_INF ################

        elif options.db_set == 'coco3d_h36m_inf':       
            
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'coco3d', 'h36m', 'mpi-inf-3dhp']
            self.dataset_dict = { 'coco3d': 0, 'h36m': 1, 'mpi-inf-3dhp':2}
            self.partition = [0.3, 0.5, 0.2]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            self.length = max([len(ds) for ds in self.datasets])

            print("sampling rate: {}".format(self.partition)) 
        
        elif options.db_set == 'cocoall3d_h36m_inf':       
            
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'cocoall3d', 'h36m', 'mpi-inf-3dhp']
            self.dataset_dict = { 'cocoall3d': 0, 'h36m': 1, 'mpi-inf-3dhp':2}
            self.partition = [0.4, 0.5, 0.1]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            self.length = max([len(ds) for ds in self.datasets])

            print("sampling rate: {}".format(self.partition)) 


        elif options.db_set == 'cocoall3d_h36m_inf_3dpw':       
            
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'cocoall3d', 'h36m', 'mpi-inf-3dhp', '3dpw_train']
            self.dataset_dict = { 'cocoall3d': 0, 'h36m': 1, 'mpi-inf-3dhp':2, '3dpw_train':3}
            # self.partition = [0.4, 0.4, 0.1, 0.1]
            self.partition = [0.3, 0.4, 0.1, 0.2]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            self.length = max([len(ds) for ds in self.datasets])

            print("sampling rate: {}".format(self.partition)) 


        elif options.db_set == 'cocoall3d_h36m_inf_posetrack3d':       #All available without 3dpw
            
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'cocoall3d', 'h36m', 'mpi-inf-3dhp', 'posetrack3d']
            self.dataset_dict = { 'cocoall3d': 0, 'h36m': 1, 'mpi-inf-3dhp':2, 'posetrack3d':3}
            self.partition = [0.4, 0.4, 0.1, 0.1]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            self.length = max([len(ds) for ds in self.datasets])

            print("sampling rate: {}".format(self.partition)) 

        elif options.db_set == 'cocoall3d_h36m_inf_3dpw_posetrack3d':       #All available including 3dpw
            
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'cocoall3d', 'h36m', 'mpi-inf-3dhp', '3dpw_train', 'posetrack3d']
            self.dataset_dict = { 'cocoall3d': 0, 'h36m': 1, 'mpi-inf-3dhp':2, '3dpw_train':3, 'posetrack3d':4}
            self.partition = [0.3, 0.4, 0.1, 0.1, 0.1]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            self.length = max([len(ds) for ds in self.datasets])

            print("sampling rate: {}".format(self.partition)) 


        elif options.db_set == 'cocoall3d_h36m_inf_posetrack3d_lspamttrain':       #All available without 3dpw
            
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'cocoall3d', 'h36m', 'mpi-inf-3dhp', 'posetrack3d','lspet3d-amt-train']
            self.dataset_dict = { 'cocoall3d': 0, 'h36m': 1, 'mpi-inf-3dhp':2, 'posetrack3d':3, 'lspet3d-amt-train':4}
            self.partition = [0.4, 0.35, 0.1, 0.1, 0.05]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            self.length = max([len(ds) for ds in self.datasets])

            print("sampling rate: {}".format(self.partition))

        elif options.db_set == 'multicrop_cocoall3d_h36m_inf_posetrack3d_lspamttrain':       #All available without 3dpw
            
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'multicrop_cocoall3d', 'h36m', 'mpi-inf-3dhp', 'multicrop_posetrack3d','lspet3d-amt-train']
            self.dataset_dict = { 'multicrop_cocoall3d': 0, 'h36m': 1, 'mpi-inf-3dhp':2, 'multicrop_posetrack3d':3, 'lspet3d-amt-train':4}
            self.partition = [0.4, 0.35, 0.1, 0.1, 0.05]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            self.length = max([len(ds) for ds in self.datasets])

            print("sampling rate: {}".format(self.partition)) 

        elif options.db_set == 'cocoall3d_h36m_inf_3dpw_posetrack3d_lspamttrain':       #All available including 3dpw
            
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'cocoall3d', 'h36m', 'mpi-inf-3dhp', '3dpw_train', 'posetrack3d', 'lspet3d-amt-train']
            self.dataset_dict = { 'cocoall3d': 0, 'h36m': 1, 'mpi-inf-3dhp':2, '3dpw_train':3, 'posetrack3d':4, 'lspet3d-amt-train':5}
            self.partition = [0.3, 0.35, 0.1, 0.1, 0.1, 0.05]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            self.length = max([len(ds) for ds in self.datasets])

            print("sampling rate: {}".format(self.partition)) 


        elif options.db_set == 'cocoall3d_h36m_inf_posetrack3d_lspamttrain_ochumantrain':       #All available without 3dpw
            
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'cocoall3d', 'h36m', 'mpi-inf-3dhp', 'posetrack3d','lspet3d-amt-train','ochuman3d']
            self.dataset_dict = { 'cocoall3d': 0, 'h36m': 1, 'mpi-inf-3dhp':2, 'posetrack3d':3, 'lspet3d-amt-train':4, 'ochuman3d':5}
            self.partition = [0.4, 0.3, 0.1, 0.1, 0.05, 0.05]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            self.length = max([len(ds) for ds in self.datasets])

            print("sampling rate: {}".format(self.partition)) 

        elif options.db_set == 'multicrop_cocoall3d_h36m_inf_posetrack3d_lspamttrain_ochumantrain':       #All available without 3dpw
            
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'multicrop_cocoall3d', 'h36m', 'mpi-inf-3dhp', 'multicrop_posetrack3d','lspet3d-amt-train','ochuman3d']
            self.dataset_dict = { 'multicrop_cocoall3d': 0, 'h36m': 1, 'mpi-inf-3dhp':2, 'multicrop_posetrack3d':3, 'lspet3d-amt-train':4, 'ochuman3d':5}
            self.partition = [0.4, 0.3, 0.1, 0.1, 0.05, 0.05]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            self.length = max([len(ds) for ds in self.datasets])

            print("sampling rate: {}".format(self.partition)) 



        elif options.db_set == 'cocoall3d_h36m_inf_3dpw_posetrack3d_lspamttrain_ochumantrain':       #All available including 3dpw
            
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'cocoall3d', 'h36m', 'mpi-inf-3dhp', '3dpw_train', 'posetrack3d', 'lspet3d-amt-train','ochuman3d']
            self.dataset_dict = { 'cocoall3d': 0, 'h36m': 1, 'mpi-inf-3dhp':2, '3dpw_train':3, 'posetrack3d':4, 'lspet3d-amt-train':5, 'ochuman3d':6}
            self.partition = [0.3, 0.3, 0.1, 0.1, 0.1, 0.05, 0.05]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            self.length = max([len(ds) for ds in self.datasets])

            print("sampling rate: {}".format(self.partition)) 

        elif options.db_set == 'cocoall3d_h36m_inf_posetrack3d_lspamtall':       #All available without 3dpw
            
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'cocoall3d', 'h36m', 'mpi-inf-3dhp', 'posetrack3d','lspet3d-amt']
            self.dataset_dict = { 'cocoall3d': 0, 'h36m': 1, 'mpi-inf-3dhp':2, 'posetrack3d':3, 'lspet3d-amt':4}
            self.partition = [0.4, 0.35, 0.1, 0.1, 0.05]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            self.length = max([len(ds) for ds in self.datasets])

            print("sampling rate: {}".format(self.partition)) 

        elif options.db_set == 'cocoall3d_h36m_inf_3dpw_posetrack3d_lspamtall':       #All available including 3dpw
            
            print(">>> Selected DBSet: {}".format(options.db_set))
            self.dataset_list = [ 'cocoall3d', 'h36m', 'mpi-inf-3dhp', '3dpw_train', 'posetrack3d', 'lspet3d-amt']
            self.dataset_dict = { 'cocoall3d': 0, 'h36m': 1, 'mpi-inf-3dhp':2, '3dpw_train':3, 'posetrack3d':4, 'lspet3d-amt':5}
            self.partition = [0.3, 0.35, 0.1, 0.1, 0.1, 0.05]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            self.length = max([len(ds) for ds in self.datasets])

            print("sampling rate: {}".format(self.partition)) 



        else:
            assert False
          
            
        assert len(self.partition) == len(self.dataset_list)
        print(">>> Total DB num: {} | total in-the-wild DB num: {}".format(total_length, length_itw))

        """
        Data distribution inside each batch:
        30% H36M - 60% ITW - 10% MPI-INF
        """
        # self.partition = [0.1, 0.2, 0.2, 0.5]
        # self.partition = [.8*len(self.datasets[0])/length_itw,
        #                   .8*len(self.datasets[1])/length_itw,
        #                   .8*len(self.datasets[2])/length_itw, 
        #                   .8*len(self.datasets[3])/length_itw, 
        #                   0.2]
        self.partition = np.array(self.partition).cumsum()

        #Save Dataset information as log
        with open(join(options.log_dir, "dbinfo.json"), "w") as f:
            json.dump(config.DATASET_FILES[1], f, indent=4)     #config.DATASET_FILES[1] has training info


    def __getitem__(self, index):

        p = np.random.rand()
        for i in range(len(self.partition)):
            if p <= self.partition[i]:
                return self.datasets[i][index % len(self.datasets[i])]

    def __len__(self):
        return self.length
