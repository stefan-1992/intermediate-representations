import os
import random

import evaluate
import numpy as np
import pandas as pd
import torch
import wandb
from anytree import findall
from datasets import Dataset
from lrml_score import compute_lrml
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from lrml_utils import (fix_lrml_tokenisation, node_to_lrml, parse_to_tree,
                        resolve_expressions, reverse_move_and_or_to_data_node,
                        reverse_resolve_expressions)

metric = None


def get_splits(tokenizer, input_series, label_series, shuffle=False, test_size=0.1, max_length=1024):
    dataset = transform_lists_to_ds(
        tokenizer, input_series.tolist(), label_series.tolist(), max_length=max_length)
    return dataset.train_test_split(test_size=test_size, shuffle=shuffle)


def get_doc_split(tokenizer, df, text, lrml, short=False):
    if short:
        train_index = df.loc[(~(df['file'].str.contains('G14VM1')) & ~(df['file'].str.contains('D1AS1')) & ~(
            df['file'].str.contains('B2AS1'))) & (df['lrml'].str.len() < 2000)].index
        valid_index = df.loc[((df['file'].str.contains('G14VM1')) | (df['file'].str.contains('D1AS1')) | (
            df['file'].str.contains('B2AS1'))) & (df['lrml'].str.len() < 372)].index
    else:
        train_index = df.loc[(~(df['file'].str.contains('G14VM1')) & ~(df['file'].str.contains('D1AS1')) & ~(
            df['file'].str.contains('B2AS1')))].index
        valid_index = df.loc[((df['file'].str.contains('G14VM1')) | (df['file'].str.contains('D1AS1')) | (
            df['file'].str.contains('B2AS1')))].index
    train_ds = transform_lists_to_ds(
        tokenizer, text[train_index].tolist(), lrml[train_index].tolist())
    valid_ds = transform_lists_to_ds(
        tokenizer, text[valid_index].tolist(), lrml[valid_index].tolist())
    return {'train': train_ds, 'test': valid_ds}


def get_file_names_from_df(df):
    return df['file'].str.split('-').apply(lambda x: x[1]).str.split('#').apply(lambda x: x[0])


def get_lrml_dataset(tokenizer, prefix='translate English to LegalRuleML: ', short=False):
    lrml_df = pd.read_csv('data/lrml/lrml_v2.csv')
    return get_doc_split(tokenizer, lrml_df, prefix + get_file_names_from_df(lrml_df) + ' ' + lrml_df['text'], fix_lrml_tokenisation(lrml_df['lrml']), short=short)


def transform_lists_to_ds(tokenizer, inputs, labels, max_length=1024):
    input_tokens = tokenizer(inputs, truncation=True, max_length=max_length)
    with tokenizer.as_target_tokenizer():
        label_tokens = tokenizer(
            labels, truncation=True, max_length=max_length)

    return Dataset.from_dict({'input_ids': input_tokens.input_ids,
                              'attention_mask': input_tokens.attention_mask, 'labels': label_tokens.input_ids
                              })


def get_tokenizer_and_model(model_path, saved_model=None):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if saved_model is not None:
        model_path = saved_model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return tokenizer, model.cuda()


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def remove_padded_values(decoded_preds, decoded_labels):
    while decoded_preds and decoded_preds[-1] == '' and decoded_labels and decoded_labels[-1] == '':
        decoded_preds.pop()
        decoded_labels.pop()
    return decoded_preds, decoded_labels


def compute_metrics(eval_preds, tokenizer, custom_postprocess=None):
    global metric
    preds, labels, *others = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    if metric is None:
        metric = evaluate.load('bleu')

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    decoded_others = [tokenizer.batch_decode(
        i, skip_special_tokens=True) for i in others]

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = remove_padded_values(
        decoded_preds, decoded_labels)

    old_labels = decoded_labels
    old_preds = decoded_preds
    # Some simple post-processing
    if custom_postprocess:
        decoded_preds, decoded_labels = custom_postprocess(
            decoded_preds, decoded_labels)
    else:
        decoded_preds, decoded_labels = postprocess_text(
            decoded_preds, decoded_labels)

    lrml_metric = compute_lrml(
        predictions=decoded_preds, references=decoded_labels, entity_weight=2, filter_empty=True)
    if sum([len(i) for i in decoded_preds]) > 0:
        lrml_metric.update(metric.compute(
            predictions=decoded_preds, references=decoded_labels))

    # wandb log as table decoded_inputs, decoded_preds, old_labels, old_preds
    lrml_metric['predictions'] = wandb.Table(data=list(zip(decoded_others[0], decoded_labels, decoded_preds, old_labels, old_preds)), columns=[
              'input', 'original label', 'original pred', 'IR label', 'IR pred'])

    index = random.randint(0, len(decoded_preds)-1)
    lrml_metric['RANDOM_PRED'] = old_preds[index]
    lrml_metric['RANDOM_LABEL'] = old_labels[index]
    lrml_metric['RANDOM_PRED_EVAL'] = decoded_preds[index]
    lrml_metric['RANDOM_LABEL_EVAL'] = decoded_labels[index]
    lrml_metric['RANDOM_INPUT'] = decoded_others[0][index]

    if len(decoded_others) > 1:
        lrml_metric['ir_change_ratio'] = len([j for i, j in enumerate(
            old_preds) if j == decoded_others[1][i]])/len(old_preds)

    return lrml_metric


def post_process_lrml(lrml, revert_and_or):
    lrml = lrml.strip()
    lrml = lrml[lrml.find('if('):]
    lrml = lrml.replace('[', '(').replace(']', ')').replace(
        '{', '(').replace('}', ')')
    lrml = lrml.replace(').', ')')
    lrml = fix_then(lrml)
    if revert_and_or:
        lrml = save_revert_and_or(lrml)
    return lrml


def fix_then(lrml):
    tree = parse_to_tree(lrml)
    if len(tree.children) == 1:
        thens = findall(tree, filter_=lambda x: ((x.name.strip() == 'then')))
        if len(thens) > 0:
            thens[0].parent = tree
    return node_to_lrml(tree)


def save_revert_and_or(lrml):
    if 'atom(' in lrml:
        lrml = reverse_move_and_or_to_data_node(lrml)
    else:
        lrml = resolve_expressions(reverse_move_and_or_to_data_node(
            reverse_resolve_expressions(lrml, fix_errors=True, prefix=' ')))
    return lrml


def get_optimizer(model, lr, weight_decay):
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)


def report_best(metrics):
    wandb.log({str(key) + '.max': val for key, val in metrics.items()})


def write_predictions(path, tokenizer, inputs, label_ids, predictions, metrics):

    os.makedirs('predictions', exist_ok=True)
    decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)
    label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
    decoded_label_ids = tokenizer.batch_decode(
        label_ids, skip_special_tokens=True)
    decoded_predictions = tokenizer.batch_decode(
        predictions, skip_special_tokens=True)
    with open('predictions/' + path, 'w') as f:
        f.write(str(metrics) + '\n\n')
        for i in range(len(decoded_inputs)):
            f.write('Scores: ' + str(compute_lrml(predictions=[decoded_predictions[i]], references=[decoded_label_ids[i]], entity_weight=2, filter_empty=True)) +
                    ';\n Input: ' + decoded_inputs[i] + ';\n Label: ' + decoded_label_ids[i] + ';\n Prediction: ' + decoded_predictions[i] + '\n\n')


def padded_concat(tokenizer, tensor_list):
    padded_tensor = torch.nested_tensor(
        tensor_list).to_padded_tensor(tokenizer.pad_token_id)
    return padded_tensor.reshape((-1, padded_tensor.shape[-1]))


def evaluate_lrml(tokenizer, model, data_loader, gen_kwargs, fp_16, custom_postprocess=None, predictions_path=None):
    model.eval()
    generated_list = []
    label_list = []
    input_list = []
    eval_progress_bar = tqdm(range(len(data_loader)), leave=False)
    for step, batch in enumerate(data_loader):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=fp_16):
                generated_tokens = model.generate(
                    batch["input_ids"].cuda(),
                    attention_mask=batch["attention_mask"].cuda(),
                    **gen_kwargs
                ).cpu()

        labels = batch["labels"].cpu()
        input_ids = batch["input_ids"].cpu()

        generated_list.append(generated_tokens)
        label_list.append(labels)
        input_list.append(input_ids)
        eval_progress_bar.update(1)

    eval_metric = compute_metrics(
        (padded_concat(tokenizer, generated_list), padded_concat(tokenizer, label_list), padded_concat(tokenizer, input_list)), tokenizer, custom_postprocess)
    if predictions_path:
        write_predictions(path=predictions_path, tokenizer=tokenizer, inputs=padded_concat(tokenizer, input_list), label_ids=padded_concat(
            tokenizer, label_list), predictions=padded_concat(tokenizer, generated_list), metrics=eval_metric)
    eval_progress_bar.close()
    print(eval_metric)
    wandb.log(eval_metric)
    return eval_metric

