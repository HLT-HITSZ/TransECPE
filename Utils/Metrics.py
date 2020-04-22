def emotion_metric(preds, grounds, documents_len):
    tn, fn, fp, tp = 0, 0, 0, 0
    for i in range(len(preds)):
        pred, ground, dl = preds[i], grounds[i], documents_len[i]
        t_emos = set([x[0] for x in ground])            
        p_emos = set([x[0] for x in pred])
        tp += len(p_emos & t_emos)
        fn += (len(t_emos) - len(p_emos & t_emos))
        fp += (len(p_emos) - len(p_emos & t_emos))
        tn += (dl - (len(t_emos) + len(p_emos) - len(p_emos & t_emos)))
    if tp + fp == 0:
        pre = tp/(tp + fp + 1e-10)
    else:
        pre = tp/(tp + fp)
    if tp + fn == 0:
        rec = tp/(tp + fn + 1e-10)
    else:
        rec = tp/(tp + fn)
    if pre == 0. and rec == 0.:
        f1 = (2 * pre *rec)/(pre + rec + 1e-10)
    else:
        f1 = (2 * pre *rec)/(pre + rec)
    acc = (tp + tn)/(tp + tn + fp +fn)
    emo_metric = (pre, rec, f1, acc)
    return emo_metric

def cause_metric(preds, grounds, documents_len):
    tn, fn, fp, tp = 0, 0, 0, 0
    for i in range(len(preds)):
        pred, ground, dl = preds[i], grounds[i], documents_len[i]
        t_cause = set([x[1] for x in ground])            
        p_cause = set([x[1] for x in pred if len(x) == 2])
        tp += len(p_cause & t_cause)
        fn += (len(t_cause) - len(p_cause & t_cause))
        fp += (len(p_cause) - len(p_cause & t_cause))
        tn += (dl - (len(t_cause) + len(p_cause) - len(p_cause & t_cause)))
    if tp + fp == 0:
        pre = tp/(tp + fp + 1e-10)
    else:
        pre = tp/(tp + fp)
    if tp + fn == 0:
        rec = tp/(tp + fn + 1e-10)
    else:
        rec = tp/(tp + fn)
    if pre == 0. and rec == 0.:
        f1 = (2 * pre *rec)/(pre + rec + 1e-10)
    else:
        f1 = (2 * pre *rec)/(pre + rec)
    acc = (tp + tn)/(tp + tn + fp +fn)
    cse_metric = (pre, rec, f1, acc)
    return cse_metric

def pair_metric(preds, grounds):
    tn, fn, fp, tp = 0, 0, 0, 0
    for i in range(len(preds)):
        pred, ground = preds[i], grounds[i]
        t_pair = set(ground)            
        p_pair = set([x for x in pred if len(x) == 2])
        tp += len(p_pair & t_pair)
        fn += (len(t_pair) - len(p_pair & t_pair))
        fp += (len(p_pair) - len(p_pair & t_pair))
    if tp + fp == 0:
        pre = tp/(tp + fp + 1e-10)
    else:
        pre = tp/(tp + fp)
    if tp + fn == 0:
        rec = tp/(tp + fn + 1e-10)
    else:
        rec = tp/(tp + fn)
    if pre == 0. and rec == 0.:
        f1 = (2 * pre *rec)/(pre + rec + 1e-10)
    else:
        f1 = (2 * pre *rec)/(pre + rec)
    pr_metric = (pre, rec, f1)
    return pr_metric