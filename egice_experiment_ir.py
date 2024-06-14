# Follows the translation setup https://github.com/huggingface/notebooks/blob/main/examples/translation.ipynb

import os
import shutil

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pandas as pd

from lrml_utils import *


dataset_name = 'lrml'

DOC_SPLIT = 'doc_split'
RANDOM_SPLIT = 'random_split'


df = pd.read_csv('data/lrml_ds_v8.csv')


def get_dataset(tokenizer, df, split_type, ds_type, simplifications):
    if ds_type > 0:
        df = df.loc[df[split_type] == ds_type]
    text, lrml = get_text_and_lrml(df, simplifications)
    return transform_lists_to_ds(tokenizer, text.tolist(), lrml.tolist())


def get_splits(tokenizer, df, split_type, simplifications):
    train_ds = get_dataset(tokenizer, df, split_type, 1, simplifications)
    valid_ds = get_dataset(tokenizer, df, split_type, 2,
                           simplifications + ['pred'])
    test_ds = get_dataset(tokenizer, df, split_type, 3,
                          simplifications + ['pred'])
    # Additional test sets
    if not 'ir' in simplifications:
        test_oracle_ds = get_dataset(
            tokenizer, df, split_type, 3, simplifications + ['oracle'])
    else:
        test_oracle_ds = None

    if not 'ir' in simplifications:
        test_no_sep_ds = get_dataset(
            tokenizer, df, split_type, 3, simplifications + ['no_sep'])
    else:
        test_no_sep_ds = None

    # Predict values for following trainings
    if 'ir' in simplifications:
        all_ds = get_dataset(tokenizer, df, split_type, 0, simplifications)
    else:
        all_ds = None
    return {'train': train_ds, 'valid': valid_ds, 'test': test_ds, 'test_oracle': test_oracle_ds, 'test_no_sep': test_no_sep_ds, 'all': all_ds}


def get_file_names_from_df(df):
    return df['file'].str.split('-').apply(lambda x: x[1]).str.split('#').apply(lambda x: x[0])


def get_text_and_lrml(df, simplifications):
    lrml = df['label']
    if 'ir' in simplifications:
        text = df['input']
    elif 'no_sep' in simplifications:
        text = df['input_no_sep']
    elif 'pred' in simplifications:
        text = df['input_pred']
    elif 'oracle' in simplifications:
        text = df['input_oracle']
    elif 'train-pred' in simplifications:
        text = df['input_pred']
    elif 'train-oracle' in simplifications:
        text = df['input_oracle']
    elif 'train-oracle-pred' in simplifications:
        text = pd.concat((df['input_pred'], df['input_oracle']))
        lrml = pd.concat((df['label'], df['label']))
    return text, lrml


def get_original(lrml, simplifications):

    return lrml


# We evaluate still on the original LRML
def clean_pred(lrml, simplifications, added_spaces=True):
    prefix = ' ' if added_spaces else ''

    # postprocessing_
    lrml = lrml.strip()
    lrml = lrml[lrml.find('if('):]
    lrml = lrml.replace('[', '(').replace(']', ')').replace(
        '{', '(').replace('}', ')')
    lrml = lrml.replace(').', ')')
    lrml = fix_then(lrml, prefix=prefix)

    # Fix errors is postprocessing
    lrml = reverse_loop(lrml, prefix=prefix)
    lrml = reverse_resolve_expressions(lrml, fix_errors=True, prefix=prefix)
    lrml = reverse_combine_rel_and_var(lrml, prefix=prefix)
    # Maybe not required?
    lrml = reverse_move_and_or_to_data_node(lrml)
    lrml = reverse_units(lrml, prefix=prefix)

    # postprocessing
    lrml = remove_duplicate_expressions(lrml, prefix + 'obligation')
    lrml = remove_duplicate_expressions(lrml, prefix + 'expression')
    return lrml


def custom_postprocess_text(preds, labels, simplifications):
    preds = [clean_pred(pred, simplifications) for pred in preds]
    labels = [[clean_pred(label, simplifications)] for label in labels]

    return preds, labels


def fix_then(lrml, prefix):
    tree = parse_to_tree(lrml)
    if len(tree.children) == 1:
        thens = findall(tree, filter_=lambda x: ((x.name == prefix + 'then')))
        if len(thens) > 0:
            thens[0].parent = tree
    return node_to_lrml(tree)


def save_split_camel(text):
    regex = re.compile(r'(?<=[a-z0-9])(?=[A-Z])')
    regex2 = re.compile(r'(?<=[a-z])\.(?=[a-z])')
    if not ' ' in text:
        text = regex.sub(' ', text).lower().replace('_', ' ')
        text = regex2.sub('. ', text).replace('_', ' ')
    else:
        text = text.replace('_', ' ')
    return text


# Unique leave nodes only
def get_ir_entities(lrml):
    nodes = [i.name for i in parse_to_tree(lrml).leaves]
    # Split at dot and add both strings to list (?=[a-z])\.|\.(?=[a-z])
    nodes = [item for sublist in [
        re.split('(?=[a-z])\.|\.(?=[a-z])', i) for i in nodes] for item in sublist]
    # Remove x + digit like x0, x1, x2
    nodes = [i for i in nodes if not re.match(r'x\d+', i)]
    # Split camel case
    nodes = [save_split_camel(i) for i in nodes]
    # Join and unique
    return ', '.join(list(dict.fromkeys(nodes)))


keywords = ['if', 'then', 'and', 'or', 'obligation',
            'permission', 'prohibition', 'not', 'loop']


def get_expressions(lrml):
    tree = parse_to_tree(lrml)
    expr_node = findall(tree, filter_=lambda x: (x.parent and x.name.strip(
    ) not in keywords and x.parent.name.strip() in keywords))
    return [node_to_lrml(i) for i in expr_node]

# Unique leave nodes only


def get_ir_expressions(lrml):
    return ','.join(get_expressions(tree_based_spacing(lrml))).strip()


EXP0 = [get_ir_entities]
EXP1 = [get_ir_expressions]

EXPERIMENTS = [EXP0, EXP1]
EXPERIMENT_NAMES = ['ir-entities', 'ir-exp']


if __name__ == '__main__':
    parser = ArgumentParser(description="Trainer script",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-mp', '--model-path', type=str, default='models/model_t5-amr/',
                        help='Model path to be used.')
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--write-predictions', action='store_true')
    parser.add_argument('--sample-data', action='store_true')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4,
                        help='The learning rate.')
    parser.add_argument('-bs', '--batchsize', type=int, default=8,
                        help='The batch size used for training. Evaluation uses batchsize * 2.')
    parser.add_argument('-nb', '--num-beams', type=int, default=3,
                        help='Beam size for evaluation')
    parser.add_argument('--seed', type=str, default='43,44,45',
                        help='The random seed. Comma separated list allowed - 43,44,45. Default: 43,44,45')
    parser.add_argument('--steps', type=int, default=500,
                        help='Evaluate after how many steps?')
    parser.add_argument('--eval', type=str, default='test',
                        help='On which dataset should be evaluated after training?')
    parser.add_argument('--exp', type=str, required=True,
                        help='The experiment number.')
    parser.add_argument('--gpu', type=int, required=True,
                        help='Which gpu should it run on')
    parser.add_argument('--epochs', type=int, default=20,
                        help='The random seed.')
    parser.add_argument('-esp', '--early-stopping-patience', type=int, default=20,
                        help='The early stopping patience.')
    parser.add_argument('-es', '--eval_strategy', type=str, default='epoch',
                        help='Strategy for evaluation: epoch or steps?')
    parser.add_argument('--skip', type=int, default=0,
                        help='Strategy for evaluation: epoch or steps?')

    args, unknown = parser.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    lr = args.learning_rate
    bs = args.batchsize


    # Original representation needs longer n-grams

    for exp in args.exp.split(','):
        exp = int(exp)
        no_repeat_ngram_size = 9
        max_length = 512
        #     Late import to be able to set CUDA_VISIBLE_DEVICES before torch is initialised
        from egice_training import transform_lists_to_ds, train, init_tokenizer

        model_name = 't5-amr' if 't5-amr' in args.model_path else args.model_path
        tokenizer_name = 't5-base' if 't5-amr' in args.model_path else args.model_path

        tokenizer = init_tokenizer(tokenizer_name)

        gen_kwargs0 = {
            "max_length": max_length,
            "num_beams": args.num_beams,
            "early_stopping": False,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": no_repeat_ngram_size,
        }
        gen_kwargss = [gen_kwargs0]

        skipped = 0

        for split in [RANDOM_SPLIT, DOC_SPLIT]:
            for seed in [int(i) for i in args.seed.split(',')]: 
                if skipped < args.skip:
                    skipped += 1
                    print('Skipping', skipped)
                    continue
                old_model_paths = {}
                for train_type in ['ir', 'train-pred', 'train-oracle', 'train-oracle-pred']:
                    if train_type == 'ir':
                        model_paths_dict = {model_name: args.model_path}
                    else:
                        model_paths_dict = {
                            model_name: args.model_path, model_name + '_ir': old_model_paths['ir']}
                    for model_name1, model_path1 in model_paths_dict.items():

                        exp_name = model_name1 + '_' + \
                            str(exp) + '_' + \
                            EXPERIMENT_NAMES[exp] + '_' + train_type + '_' + split + '_esp' + str(
                                args.early_stopping_patience) + '_e' + str(args.epochs) + '_' + str(seed)

                        if train_type == 'ir':
                            if exp == 0:
                                input_prefix = 'extract LegalRuleML entities: ' + \
                                    get_file_names_from_df(df) + ' '
                                df['label'] = df['lrml'].apply(get_ir_entities)
                                metric_for_best_model = 'bleu'
                            elif exp == 1:
                                input_prefix = 'extract LegalRuleML expressions: ' + \
                                    get_file_names_from_df(df) + ' '
                                df['label'] = df['lrml'].apply(
                                    get_ir_expressions)
                                metric_for_best_model = 'lrml_f_score'
                            elif exp == 2:
                                input_prefix = 'translate English to simplified LegalRuleML: ' + \
                                    get_file_names_from_df(df) + ' '

                            df['input'] = input_prefix + \
                                get_file_names_from_df(df) + ' ' + df['text']

                            post_process = None

                        # Only for first training required
                        elif all_predictions is not None:
                            input_prefix = 'translate English to LegalRuleML: ' + \
                                get_file_names_from_df(df) + ' '
                            df['input_no_sep'] = input_prefix + df['text']
                            df['input_pred'] = input_prefix + df['text'] + \
                                '<sep>' + all_predictions[:len(df)]
                            if exp == 0:
                                df['input_oracle'] = input_prefix + df['text'] + \
                                    '<sep>' + df['lrml'].apply(get_ir_entities)
                            elif exp == 1:
                                df['input_oracle'] = input_prefix + df['text'] + \
                                    '<sep>' + df['lrml'].apply(get_ir_expressions)
                            df['label'] = df['lrml'].apply(
                                lambda x: get_original(x, [])).apply(tree_based_spacing)
                            metric_for_best_model = 'lrml_f_score'

                            def post_process(x, y): return custom_postprocess_text(
                                x, y, simplifications=[])

                        ds = get_splits(tokenizer, df, split, [train_type])
                        hyperparameters = dict(epochs=args.epochs, lr=lr, bs=bs, adafactor=False, weight_decay=1.000e-1, metric_for_best_model=metric_for_best_model,
                                               num_warmup_steps=0, dataset=dataset_name, model_name=model_name1, model_path=[model_path1],
                                               post_process=post_process, lr_scheduler_type='linear', gradient_accumulation_steps=1,
                                               run_name=exp_name, eval_strategy=args.eval_strategy, eval_steps=args.steps,
                                               early_stopping_threshold=args.early_stopping_patience, max_length=max_length,
                                               seed=seed, write_predictions=args.write_predictions,
                                               evaluate_only=args.evaluate_only, evaluation_set=args.eval, runs=1, start_epoch=0, skip_evals=0,
                                               teacher_forcing_percentage=0.0, label_augmentation=0.0, retrain=True, mask_percentage=0.0, is_ir=False,
                                               fp16=False, split=split, experiment=EXPERIMENTS[exp], exp_num=exp)

                        hyperparameters.update(
                            {k: v for k, v in gen_kwargs0.items()})

                        print(hyperparameters)

                        last_model_path, all_predictions, train_predictions = train(datasets=ds, hyperparameters=hyperparameters,
                                                                 delete_model=train_type != 'ir', gen_kwargss=gen_kwargss, post_process=post_process)
                        old_model_paths[train_type] = last_model_path

                shutil.rmtree(old_model_paths['ir'])
