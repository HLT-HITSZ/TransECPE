# encoding: utf-8
# @author: ChuangFan
# email: fanchuanghit@gmail.com


import copy


action2id = {'shift': 0, 'left_arc_ln': 1, 'left_arc_lt': 2, 
             'right_arc_ln': 3, 'right_arc_lt': 4}

def GetAction(s_pair, pairs, action2id):
    actions = []
    emotions, _ = zip(*pairs)
    emotions = set(emotions)
    if s_pair[0] in emotions and s_pair[1] in emotions:
        actions.append(action2id['right_arc_ln'])
    
    
    for pair in pairs:
        emotion = pair[0]
        cause = pair[1]
        if emotion == s_pair[0] and cause == s_pair[1]:
            actions.append(action2id['left_arc_lt'])
        elif emotion == s_pair[0] and cause != s_pair[1]:
            actions.append(action2id['left_arc_ln'])
        elif emotion == s_pair[1] and cause == s_pair[0]:
            actions.append(action2id['right_arc_lt'])
        elif emotion == s_pair[1] and cause != s_pair[0]:
            actions.append(action2id['right_arc_ln'])
    if len(actions) == 0:
        return action2id['shift']
    elif len(actions) == 1:
        return actions[0]
    else:
        if action2id['right_arc_lt'] in set(actions):
            return action2id['right_arc_lt']
        elif action2id['left_arc_lt'] in set(actions):
            return action2id['left_arc_lt']
        else:
            return actions[0]
        
def ActionSequence(text, pairs, action2id):
    stack = []
    clauses = text.split('\x01')
    buffer = list(range(len(clauses)))
    # initial state
    stack.append(0), stack.append(1)
    buffer.remove(0), buffer.remove(1)
    # the length of the shortest document in corpus is 2
    actions = []
    while len(buffer) > 0: 
        if len(stack) < 2:
            stack.append(buffer.pop(0))
        s_pair = (stack[-2], stack[-1])
        action = GetAction(s_pair, pairs, action2id)
        actions.append((copy.deepcopy(stack), copy.deepcopy(buffer), action))
        if action == action2id['shift']:
            if len(buffer) > 0:
                stack.append(buffer.pop(0))
                
        if action == action2id['right_arc_ln'] or action == action2id['right_arc_lt']:
            stack.pop(-2)
            
        if action == action2id['left_arc_ln']:
            if len(buffer) > 0:
                stack.append(buffer.pop(0))
                
        if action == action2id['left_arc_lt']:
            stack.pop(-1)

    while len(stack) >= 2:
        s_pair = (stack[-2], stack[-1])
        action = GetAction(s_pair, pairs, action2id)
        actions.append((copy.deepcopy(stack), copy.deepcopy(buffer), action))
        if action == action2id['right_arc_ln'] or action == action2id['right_arc_lt']:
            stack.pop(-2)
        elif action == action2id['left_arc_ln'] or action == action2id['left_arc_lt']:
            stack.pop(-1)
        else:
            break
         
    unique_actions = []
    for at in actions:
        if at not in unique_actions:
            unique_actions.append(at)
    return unique_actions


def Text2ActionSequence(data):
    action_sequence = []
    for i in range(len(data[0])):
        action = ActionSequence(data[0][i], data[1][i], action2id)
        action_sequence.append(action)   
    return action_sequence

def Text2SingleLabel(data):
    documents = data[0]
    pairs = data[1]
    single_labels = []
    for i in range(len(documents)):
        document = documents[i]
        dl = len(document.split('\x01'))
        pair = pairs[i]
        temp = [0] * dl
        same = []
        for pr in pair:
            if pr[0] == pr[1]:
                same.append(pr[0])
        for sm in same:
            temp[sm] = 1
        single_labels.append(temp)
    return single_labels
