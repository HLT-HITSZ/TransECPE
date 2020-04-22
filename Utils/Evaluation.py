# encoding: utf-8
# @author: ChuangFan
# email: fanchuanghit@gmail.com

from Metrics import emotion_metric, cause_metric, pair_metric

def merge_tuple_single(single_preds, tuple_preds, document_list):
    dls = [len(x.split('\x01')) for x in document_list]
    start = 0
    convert_single_label = []
    for dl in dls:
        end = start + dl
        sp = single_preds[start: end]
        convert_single_label.append([(i, i) for i in range(dl) if sp[i] == 1])
        start = end
    merge_label_list = [tuple_preds[i] + convert_single_label[i] for i in range(len(dls))]
    return merge_label_list

def EvaluationTrans(trans_model, base_encoder, data, batch_size):   
    data_len = len(data[0])
    documents_len = [len(x.split('\x01')) for x in data[0]]
    grounds = data[1]
    batch_i = 0
    trans_model.eval()
    base_encoder.eval()
    single_predicts, tuple_predicts = [], []
    while batch_i * batch_size < data_len:
        start, end = batch_i * batch_size, (batch_i +1) * batch_size
        document_list = data[0][start: end]
        pooled, clause_state_list = base_encoder(document_list)
        single_preds, tuple_preds = trans_model(pooled, None, clause_state_list, None, 'eval')
        single_predicts.extend(single_preds)
        tuple_predicts.extend(tuple_preds)
        batch_i += 1
        
    final_preds = merge_tuple_single(single_predicts, tuple_predicts, data[0])
    emo_metric = emotion_metric(final_preds, grounds, documents_len)
    cse_metric = cause_metric(final_preds, grounds, documents_len)
    pr_metric = pair_metric(final_preds, grounds)
    
    return (emo_metric, cse_metric, pr_metric)
