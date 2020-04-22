# encoding: utf-8
# @author: ChuangFan
# email: fanchuanghit@gmail.com


class Config(object):

    def __init__(self):
        
# >>>>>>>>>>>>>>>>>>>> For path <<<<<<<<<<<<<<<<<<<< #
        self.doc2pair_path = './Data/doc2pair.pkl'
        self.datasplit_path = './Data/DataSplits'
        self.bert_path = './Data/bert-base-chinese'
# >>>>>>>>>>>>>>>>>>>> For training <<<<<<<<<<<<<<<<<<<< #        
        self.seed = 1024
        self.batch_size = 3  
        self.epochs = 10
        self.showtime = 200 
        self.base_encoder_lr = 1e-5 
        self.finetune_lr = 1e-3
        self.warm_up = 5e-2
        self.weight_decay = 1e-5
        self.early_num = 3  
        ## prefix: /no_action_', /no_buffer_', /no_distance_', '/no_LSTM_', '/no_reversal_'
        self.prefix = '/with_reversal_'
# >>>>>>>>>>>>>>>>>>>> For model <<<<<<<<<<<<<<<<<<<< #   
        self.cell_size = 128  
        self.layers = 1      
        self.is_bi = True
        self.bert_output_size = 768 
        self.mlp_size = 256   
        self.operation_type = 'concatenate'
        self.scale_factor = 2
        self.dropout = 0.5
        self.max_document_len = 100
        self.position_ebd_dim = 128  # 128
        self.action_ebd_dim = 128  # 128
        ## We define 6 types actions in this paper, the CA action is separated from other actions, see paper for details. 
        ## Here, we need add two auxiliary types to denote the initial state and padding state, so we have total 6-1+2=7 
        ## action types. Specifically, 0:shift, 1:left_arc_ln, 2: left_arc_lt, 3:right_arc_ln, 4:right_arc_lt, 5:initial state
        ## 6:padding state
        self.action_type_num = 7
        ## the real action types 7-2=5
        self.label_num = 5
        self.position_trainable = False
        self.action_trainable = False
        
        