import re


def camelToWords(camel):
    camel = camel.replace('_', ' _ ')
    camel = camel.replace(',', ' , ')
    words = ' '.join(re.split(r'((?<=[a-z0-9])(?=[A-Z%]))', camel)).lower().replace('  ', ' ')
    words = ' '.join(re.split(r'((?<=[0-9])(?=[a-z]))', words)).lower().replace('  ', ' ')
    #     Fix split of kN
    words = words.replace(' k n ', ' kn ')
    
    words = re.subn(r'(?<=[a-z])\.(?=[a-z])', ' ', words)
    return words[0].replace(' _ ', ' ').split(' ')


def parse_logic_string(logic_string):
    current_text = ''
    parent_term = {'terms': [], 'text': []}
    term_stack = [parent_term]
    for i in logic_string:
        if i == '(':
            if current_text == 'then' and len(term_stack) > 1:
                term_stack = close_stack(term_stack)
            term_stack.append({'relation': current_text, 'terms': [], 'text': []})
            current_text = ''
        elif i == ')':
            if len(term_stack) > 1:
                closed_term = term_stack.pop()
                current_text = current_text.strip()
                if current_text:
                    closed_term['text'].extend(camelToWords(current_text))
                    current_text = ''
                term_stack[-1]['terms'].append(closed_term)
                term_stack[-1]['text'].extend(closed_term['text'])
        elif i != ',':
            current_text += i
    if len(term_stack) > 1:
        term_stack = close_stack(term_stack)
    return term_stack[0]


def close_stack(stack):
    while len(stack) > 1:
        closed_term = stack.pop()
        stack[-1]['terms'].append(closed_term)
        stack[-1]['text'].extend(closed_term['text'])
    return stack


def flatten(terms):
    flattened = []
    for term in terms:
        flattened.append({'relation': term['relation'], 'text': term['text'], 'is_leaf': len(term['terms']) == 0,
                          'is_entity': False})
        flattened.extend(flatten(term['terms']))
    return flattened


def flatten_logic_elements(logic_elements):
    flattened = flatten(logic_elements['terms'])
    #     Append leaf nodes separately as entities
    try:
        for i in flattened:
            if i['is_leaf']:
                flattened.append({'relation': '', 'text': i['text'], 'is_entity': True})
    except:
        pass
    return flattened


def num_contained(l1, l2):
    l1, l2 = l1.copy(), l2.copy()
    count = 0
    for i in l1:
        if i in l2:
            count += 1
            l2.remove(i)
    return count

# Used to calculate either precision or recall, based on what is entered as el1 and el2
def compare_flattened(el1, el2, entity_weight):
    el1, el2 = el1.copy(), el2.copy()
    score = 0
    additional_scores = 0
    for i in el1:
        # Predicates/Relations are only scored if they are the correct keyword. For entities relation is empty.
        options = [i2 for i2 in el2 if i2['relation'] == i['relation']]
        if len(options) >= 1:
            ext_option = max([(num_contained(i['text'], op['text']), op) for op in options], key=lambda x: x[0])
            most_similar = ext_option if ext_option[0] > 0 else {}
        else:
            most_similar = {}
        # Not match -> 0 Score.
        if most_similar:
            if most_similar[1]['is_entity']:
                # The entity weighting is done per score. Since this results in up to 2 score points per entity,
                # we keep track of those extra scores and add them to the max possible scores.
                additional_scores += entity_weight - 1
                score += most_similar[0] / len(i['text']) * entity_weight
            else:
                score += most_similar[0] / len(i['text'])
            el2.remove(most_similar[1])

    additional_scores = len([i for i in el1 if i['is_entity']]) * (entity_weight - 1)
    element_count = (len(el1) + additional_scores)

    return element_count if element_count == 0 else score / element_count


def compute_lrml(predictions, references, entity_weight, filter_empty):
    assert (len(predictions) == len(references))
    assert (len(references) > 0)
    if type(references[0]) == list:
        references = [i[0] for i in references]
    scores = []
    # For each ground truth / prediction tuple calculate Recall/Precision
    for i in range(len(predictions)):
        scores.append(compute_lrml_score(predictions[i], references[i], entity_weight, filter_empty))
    precision_new = sum([i['precision'] for i in scores]) / (len(scores))
    recall_new = sum([i['recall'] for i in scores]) / (len(scores))

    if precision_new == 0 or recall_new == 0:
        f_score = 0
    else:
        f_score = (2 * precision_new * recall_new) / (precision_new + recall_new)
    return {'lrml_f_score': f_score * 100, 'lrml_recall': recall_new * 100, 'lrml_precision': precision_new * 100}


def compute_lrml_score(predicted, label, entity_weight, filter_empty):
    # Remove clue
    if predicted.find(':') < 30:
        predicted = predicted[predicted.find(':') + 2:]
    if label.find(':') < 30:
        label = label[label.find(':') + 2:]

    pred_elements = parse_logic_string(predicted)
    truth_elements = parse_logic_string(label)
    f_pred = flatten_logic_elements(pred_elements)
    if filter_empty:
        f_pred = [i for i in f_pred if not (i['relation'].strip() == '' and i['text'] == [])]
    f_truth = flatten_logic_elements(truth_elements)
    # Overlap is compared to the f_truth length. How many of the ground truth elements were in the predicted elements? 
    recall = compare_flattened(f_truth, f_pred, entity_weight)
    # Overlap is compared to the f_pred length. How many of the predicted words were actually in the ground truth?
    precision = compare_flattened(f_pred, f_truth, entity_weight)
    return {'recall': recall, 'precision': precision}