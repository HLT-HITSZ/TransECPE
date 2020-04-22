# encoding: utf-8
# @author: ChuangFan
# email: fanchuanghit@gmail.com

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import sys
import pickle
sys.path.append('./Utils')
from Transform import action2id
from pytorch_pretrained_bert import BertModel, BertTokenizer

class BertEncoder(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_path)
    
    def padding_and_mask(self, ids_list):
        max_len = max([len(x) for x in ids_list])
        mask_list = []
        ids_padding_list = []
        for ids in ids_list:
            mask = [1.] * len(ids) + [0.] * (max_len - len(ids))
            ids = ids + [0] * (max_len - len(ids))
            mask_list.append(mask)
            ids_padding_list.append(ids)
        return ids_padding_list, mask_list
        
    def forward(self, document_list):
        text_list, tokens_list, ids_list = [], [], []
        ## The clauses in each document are splited by '\x01'
        document_len = [len(x.split('\x01')) for x in document_list]
        
        for document in document_list:
            text_list.extend(document.strip().split('\x01'))  
        for text in text_list:
            text = ''.join(text.split())
            tokens = self.tokenizer.tokenize(text)
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            tokens_list.append(tokens)
        for tokens in tokens_list:
            ids_list.append(self.tokenizer.convert_tokens_to_ids(tokens))
                
        ids_padding_list, mask_list = self.padding_and_mask(ids_list)
        ids_padding_tensor = torch.LongTensor(ids_padding_list).cuda()
        mask_tensor = torch.tensor(mask_list).cuda()
        
        _, pooled = self.bert(ids_padding_tensor, attention_mask = mask_tensor, output_all_encoded_layers=False)
        
        start = 0
        clause_state_list = []
        for dl in document_len:
            end = start + dl
            clause_state_list.append(pooled[start: end])
            start = end
        return pooled, clause_state_list

# with action reversal    
class TransitionModel(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.is_bi = config.is_bi
        self.bert_output_size = config.bert_output_size
        self.mlp_size = config.mlp_size
        self.cell_size = config.cell_size
        self.operation_type = config.operation_type
        self.scale_factor = config.scale_factor
        self.dropout = config.dropout
        self.layers = config.layers
        self.max_document_len = config.max_document_len
        self.position_ebd_dim = config.position_ebd_dim
        self.position_embedding = nn.Embedding(self.max_document_len-1, self.position_ebd_dim)
        self.position_trainable = config.position_trainable
        self.action_ebd_dim = config.action_ebd_dim
        self.action_type_num = config.action_type_num
        self.action_embedding = nn.Embedding(self.action_type_num, self.action_ebd_dim) 
        self.action_trainable = config.action_trainable
        self.label_num = config.label_num
        self.stack_cell = nn.LSTM(self.bert_output_size, self.cell_size, self.layers, bidirectional=self.is_bi)
        self.buffer_cell = nn.LSTM(self.bert_output_size, self.cell_size, self.layers, bidirectional=self.is_bi)
        self.action_cell = nn.LSTM(self.action_ebd_dim, self.cell_size, self.layers, bidirectional=False)
            
        if self.operation_type == 'attention':
            self.attention_layer = nn.Sequential(
                nn.Linear(self.bert_output_size, self.hidden_size),
                nn.Linear(self.hidden_size, 1)
            )
        ## The classifier for the CA action
        self.single_MLP = nn.Sequential(
            nn.Linear(self.bert_output_size, self.mlp_size),
            nn.BatchNorm1d(self.mlp_size),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.mlp_size, self.mlp_size//self.scale_factor),
            nn.BatchNorm1d(self.mlp_size//self.scale_factor),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.mlp_size//self.scale_factor, 2)
        )
        ## The classifier for the other actions
        self.tuple_MLP = nn.Sequential(
            nn.Linear(self.cell_size*2*2+self.cell_size+self.position_ebd_dim, self.mlp_size),
            nn.BatchNorm1d(self.mlp_size),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.mlp_size, self.mlp_size//self.scale_factor),
            nn.BatchNorm1d(self.mlp_size//self.scale_factor),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.mlp_size//self.scale_factor, self.label_num)
        )
        
        self.init_weight()
        
    def init_weight(self):
        for name, param in self.named_parameters():
            if name.find("weight") != -1:
                if len(param.data.size()) > 1:
                    nn.init.xavier_normal(param.data)
                else:
                    param.data.uniform_(-0.1, 0.1)
            elif name.find("bias") != -1:
                param.data.uniform_(-0.1, 0.1)
            else:
                continue
        self.position_embedding.weight.requires_grad = self.position_trainable
        self.action_embedding.weight.requires_grad = self.action_trainable
    
    def init_hidden(self, batch_size, mode):
        if mode == 'action':
            hidden = [Variable(torch.zeros(self.layers, batch_size, self.cell_size).cuda()),
                      Variable(torch.zeros(self.layers, batch_size, self.cell_size).cuda())
                     ]
        else:
            if self.is_bi:
                hidden = [Variable(torch.zeros(self.layers*2, batch_size, self.cell_size).cuda()),
                          Variable(torch.zeros(self.layers*2, batch_size, self.cell_size).cuda())
                         ]
            else:
                hidden = [Variable(torch.zeros(self.layers, batch_size, self.cell_size).cuda()),
                          Variable(torch.zeros(self.layers, batch_size, self.cell_size).cuda())
                         ]
        return hidden
    
    def operation(self, state_1, state_2, state_3):
        if self.operation_type == 'concatenate':
            inputs = torch.cat([state_1, state_2, state_3])
        elif self.operation_type == 'mean':
            inputs = (state_1 + state_2 + state_3)/3.
        elif self.operation_type == 'sum':
            inputs = state_1 + state_2 + state_3
        elif self.operation_type == 'attention':
            stack_state = torch.stack([state_1, state_2, state_3])
            attention_logits = self.attention_layer(stack_state)
            attention_weights = F.softmax(attention_logits, 0)
            inputs = stack_state.t().mm(attention_weights).squeeze(1)
        else:
            print ('operation type error!')
        return inputs
    
    def action_encoder(self, action_sequence_list):
        action_list = [[x[-1] for x in asl] for asl in action_sequence_list]
        action_len_list = [len(x) for x in action_list]
        max_action_len = max(action_len_list)        
        action_padding_list = [[5]+x[:-1]+[6]*(max_action_len-len(x)) for x in action_list]
        action_padding_tensor = torch.tensor(action_padding_list).cuda()
        
        inputs = self.action_embedding(action_padding_tensor).permute(1, 0, 2)
        bs = inputs.size()[1]
        init_state = self.init_hidden(bs, 'action')
        outputs, _ = self.action_cell(inputs, init_state)
        outputs_permute = outputs.permute(1, 0, 2)
        output_list = [outputs_permute[i][:al] for i, al in enumerate(action_len_list)]
        output_stack = torch.cat(output_list)
        return output_stack    
    
    def reversal_sample(self, sk_1, sk_2, action):
        sk_1_forward, sk_1_backward = sk_1.chunk(2)
        sk_2_forward, sk_2_backward = sk_2.chunk(2)
        ori_sk_1, ori_sk_2, ori_act = sk_1_forward, sk_2_forward, action
        rev_sk_1, rev_sk_2 = sk_2_backward, sk_1_backward
        if action == action2id['shift']:
            rev_act = action2id['shift']
        elif action == action2id['right_arc_ln']:
            rev_act = action2id['left_arc_ln']   
        elif action == action2id['right_arc_lt']:
            rev_act = action2id['left_arc_lt']
        elif action == action2id['left_arc_ln']:
            rev_act = action2id['right_arc_ln']
        elif action == action2id['left_arc_lt']:
            rev_act = action2id['right_arc_lt']
        return ori_sk_1, ori_sk_2, ori_act, rev_sk_1, rev_sk_2, rev_act    
    
    def train_mode(self, clause_state_list, action_sequence_list):
        tuple_labels_list, distance_list = [], []
        sk_input_list, bf_input_list, sk_len_list, bf_len_list = [], [], [], [] 
        for d_i in range(len(clause_state_list)):
            clause_state, action_sequence = clause_state_list[d_i], action_sequence_list[d_i]
            for a_s in action_sequence:
                stack, buffer, action = a_s[0], a_s[1], a_s[2]
                tuple_labels_list.append(action)
                stack_input = torch.stack([clause_state[s] for s in stack])
                sk_len_list.append(stack_input.size()[0])
                sk_input_list.append(stack_input)
                distance_list.append(int(abs(stack[-2] - stack[-1])))
                if len(buffer) > 0:
                    buffer_input = torch.stack([clause_state[b] for b in buffer])
                else:
                    if stack[-1] < clause_state.size()[0]-1:
                        buffer_input = torch.stack([clause_state[stack[-1]+1]])
                    else:
                        buffer_input = torch.stack([clause_state[stack[-1]]])
                bf_input_list.append(buffer_input)
                bf_len_list.append(buffer_input.size()[0])
        max_sk_len, max_bf_len = max(sk_len_list), max(bf_len_list)
        tmp_sk_list, tmp_bf_list = [], []
        for sk_input, bf_input in zip(sk_input_list, bf_input_list):
            sk_row, sk_column = sk_input.size()
            bf_row, bf_column = bf_input.size()
            sk_tmp = Variable(torch.zeros(max_sk_len, sk_column).cuda())
            bf_tmp = Variable(torch.zeros(max_bf_len, bf_column).cuda())
            sk_tmp[:sk_row] = sk_input
            bf_tmp[:bf_row] = bf_input
            tmp_sk_list.append(sk_tmp)
            tmp_bf_list.append(bf_tmp)
        sk_input_tensor, bf_input_tensor = torch.stack(tmp_sk_list).permute(1,0,2), torch.stack(tmp_bf_list).permute(1,0,2)
        sk_bs, bf_bs = sk_input_tensor.size()[1], bf_input_tensor.size()[1]
        sk_init, bf_init = self.init_hidden(sk_bs, 'else'), self.init_hidden(bf_bs, 'else')
        sk_output, _ = self.stack_cell(sk_input_tensor, sk_init)
        bf_output, _ = self.buffer_cell(bf_input_tensor, bf_init)
        sk_output_permute, bf_output_permute = sk_output.permute(1,0,2), bf_output.permute(1,0,2)
        del sk_output
        del bf_output
        sk_update_list = [sk_output_permute[i][:sk_len] for i,sk_len in enumerate(sk_len_list)]
        bf_update_list = [bf_output_permute[i][:bf_len] for i,bf_len in enumerate(bf_len_list)]
        final_inputs_list, final_labels_list, final_distance_list, final_action_output = [], [], [], []
        inx = 0
        action_output = self.action_encoder(action_sequence_list)
        for sk_update, bf_update in zip(sk_update_list, bf_update_list):
            action = tuple_labels_list[inx]
            ori_sk_1, ori_sk_2, ori_act, rev_sk_1, rev_sk_2, rev_act = self.reversal_sample(sk_update[-2], sk_update[-1], action)
            ori_inputs = self.operation(ori_sk_1, ori_sk_2, bf_update[0])
            final_inputs_list.append(ori_inputs)
            final_labels_list.append(ori_act)
            final_distance_list.append(distance_list[inx])
            final_action_output.append(action_output[inx])
            rev_inputs = self.operation(rev_sk_1, rev_sk_2, bf_update[0])
            final_inputs_list.append(rev_inputs)
            final_labels_list.append(rev_act)
            final_distance_list.append(distance_list[inx])
            final_action_output.append(action_output[inx])
            inx += 1
        del sk_update_list
        del bf_update_list
        distance_tensor = torch.tensor(final_distance_list).cuda()
        pos_embedding = self.position_embedding(distance_tensor)
        tuple_inputs_tensor = torch.cat([torch.stack(final_inputs_list), torch.stack(final_action_output), pos_embedding], 1)

        tuple_labels_tensor = torch.LongTensor(final_labels_list).cuda()
        tuple_logits = self.tuple_MLP(tuple_inputs_tensor)
        
        return tuple_logits, tuple_labels_tensor

    def predict_action(self, state, stack, buffer, action, act_hidden): 
        stack_input = torch.stack([state[s] for s in stack]).unsqueeze(0)
        sk_init_state = self.init_hidden(1, 'else')
        sk_output, _ = self.stack_cell(stack_input.permute(1, 0, 2), sk_init_state)
        sk_output_permute = sk_output.permute(1, 0, 2).squeeze(0)
        if len(buffer) > 0:
            buffer_input = torch.stack([state[b] for b in buffer]).unsqueeze(0)
        else:
            if stack[-1] < state.size()[0]-1:
                buffer_input = torch.stack([state[stack[-1]+1]]).unsqueeze(0)
            else:
                buffer_input = torch.stack([state[stack[-1]]]).unsqueeze(0)
        bf_init_state = self.init_hidden(1, 'else')
        bf_output, _ = self.buffer_cell(buffer_input.permute(1, 0, 2), bf_init_state)
        bf_output_permute = bf_output.permute(1, 0, 2).squeeze(0)
        
        act_input = self.action_embedding(torch.tensor([[action]]).cuda())
        act_output, act_hidden = self.action_cell(act_input, act_hidden)
        act_output_permute = act_output.squeeze(0).squeeze(0)
        
        change_1_forward, change_1_backward = sk_output_permute[-2].chunk(2)
        change_2_forward, change_2_backward = sk_output_permute[-1].chunk(2)
        
        c_inputs = self.operation(change_1_forward, change_2_forward, bf_output_permute[0])
        distance = torch.tensor(int(abs(stack[-2] - stack[-1]))).cuda()
        pos_embedding = self.position_embedding(distance)
        inputs = torch.cat([c_inputs, act_output_permute, pos_embedding]).unsqueeze(0)
        tuple_logits = self.tuple_MLP(inputs)
        tuple_probs = F.softmax(tuple_logits, 1)
        action = tuple_probs.argmax(1).data.cpu().numpy()[0]

        return action, act_hidden
    
    def eval_mode(self, clause_state_list):
        predicts = []
        batch_size = len(clause_state_list)
        for d_i in range(batch_size):
            preds, stack = [], []
            document_len = clause_state_list[d_i].size()[0]
            buffer = list(range(document_len))
            stack.append(0), stack.append(1)
            buffer.remove(0), buffer.remove(1)
            state = clause_state_list[d_i]
            action = 5
            act_hidden = self.init_hidden(1, 'action')
            while len(buffer) > 0:
                if len(stack) < 2:
                    stack.append(buffer.pop(0))
                action, act_hidden = self.predict_action(state, stack, buffer, action, act_hidden)
                if action == action2id['shift']:
                    if len(buffer) > 0:
                        stack.append(buffer.pop(0))
                elif action == action2id['right_arc_ln']:
                    preds.append((stack[-1],))
                    stack.pop(-2)
                elif action == action2id['right_arc_lt']:
                    preds.append((stack[-1], stack[-2]))
                    stack.pop(-2)
                elif action == action2id['left_arc_ln']:
                    preds.append((stack[-2],))
                    if len(buffer) > 0:
                        stack.append(buffer.pop(0))
                else:  #left_arc_lt
                    preds.append((stack[-2], stack[-1]))
                    stack.pop(-1)

            while len(stack) >= 2:
                action, act_hidden = self.predict_action(state, stack, buffer, action, act_hidden)

                if action == action2id['right_arc_ln']:
                    preds.append((stack[-1],))
                    stack.pop(-2)
                elif action == action2id['right_arc_lt']:
                    preds.append((stack[-1], stack[-2]))
                    stack.pop(-2)
                elif action == action2id['left_arc_ln']:
                    preds.append((stack[-2],))
                    stack.pop(-1)
                elif action == action2id['left_arc_lt']:
                    preds.append((stack[-2], stack[-1]))
                    stack.pop(-1)
                else:
                    break
                                      
            unique_preds = []
            for pd in preds:
                if pd not in unique_preds:
                    unique_preds.append(pd)
            predicts.append(unique_preds)
            
        return predicts
    
    def forward(self, pooled, single_labels_list, clause_state_list, action_sequence_list, mode):            
        if mode == 'train':
            single_logits = self.single_MLP(pooled)
            single_labels_tensor = torch.tensor([i for x in single_labels_list for i in x]).cuda()
            tuple_logits, tuple_labels_tensor = self.train_mode(clause_state_list, action_sequence_list)
            return single_logits, single_labels_tensor, tuple_logits, tuple_labels_tensor
        elif mode == 'eval':
            single_logits = self.single_MLP(pooled)
            single_preds = list(F.softmax(single_logits, 1).argmax(1).data.cpu().numpy())
            tuple_preds = self.eval_mode(clause_state_list)
            return single_preds, tuple_preds
        else:
            print ('mode error!')

            
