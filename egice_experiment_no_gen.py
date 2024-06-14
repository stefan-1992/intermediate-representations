# Follows the translation setup https://github.com/huggingface/notebooks/blob/main/examples/translation.ipynb

import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pandas as pd

from lrml_utils import *


dataset_name = 'lrml'

DOC_SPLIT = 'doc_split'
RANDOM_SPLIT = 'random_split'

EXP_POSTPROCESS = 'postprocess'
EXP_DUPLICATES = 'duplicates'
EXP_LOOP = 'loop'
EXP_EXPRESSION = 'expression'
EXP_ATOM = 'atom'
EXP_DATA_AND_OR = 'and_or'
EXP_UNIT = 'unit'


df = pd.read_csv('data/lrml_ds_v8.csv')


def get_dataset(tokenizer, df, split_type, ds_type, simplifications):
    df = df.loc[df[split_type] == ds_type]
    text, lrml = get_text_and_lrml(df, simplifications)
    return transform_lists_to_ds(tokenizer, text.tolist(), lrml.tolist())


def get_splits(tokenizer, df, split_type, simplifications):
    train_ds = get_dataset(tokenizer, df, split_type, 1, simplifications)
    valid_ds = get_dataset(tokenizer, df, split_type, 2, simplifications)
    test_ds = get_dataset(tokenizer, df, split_type, 3, simplifications)
    return {'train': train_ds, 'valid': valid_ds, 'test': test_ds}


def get_file_names_from_df(df):
    return df['file'].str.split('-').apply(lambda x: x[1]).str.split('#').apply(lambda x: x[0])


def get_text_and_lrml(df, simplifications):
    text = 'translate English to LegalRuleML: ' + \
        get_file_names_from_df(df) + ' ' + df['text']
    lrml = df['lrml'].apply(lambda x: get_original(
        x, simplifications)).apply(tree_based_spacing)
    return text, lrml


EXP0 = []

EXPERIMENTS = [EXP0]
EXPERIMENT_NAMES = ['base_no_gen']


# We are starting from a LRML with all Simplifications.
# So Either we have to reverse the simplification in here or we have to do it during posprocessing.
# Simplest case we reverse all here (i.e. original LRML) and none in postprocessing --- Empty Simplification
# Second case we reverse none here and all in postprocessing, i.e. we test the new version. --- All Simplifications
def get_original(lrml, simplifications):
    prefix = ''
    if not EXP_LOOP in simplifications:
        lrml = reverse_loop(lrml, prefix=prefix)
    if not EXP_EXPRESSION in simplifications:
        lrml = reverse_resolve_expressions(
            lrml, fix_errors=True, prefix=prefix)
    if not EXP_ATOM in simplifications:
        lrml = reverse_combine_rel_and_var(lrml, prefix=prefix)
    if not EXP_DATA_AND_OR in simplifications:
        if 'atom(' in lrml:
            lrml = reverse_move_and_or_to_data_node(lrml)
        else:
            lrml = resolve_expressions(reverse_move_and_or_to_data_node(
                reverse_resolve_expressions(lrml, fix_errors=True, prefix=prefix)))
    if not EXP_UNIT in simplifications:
        lrml = reverse_units(lrml, prefix=prefix)
    return lrml


def clean_pred(lrml, simplifications, added_spaces=True):
    prefix = ' ' if added_spaces else ''

    # postprocessing
    if EXP_POSTPROCESS in simplifications:
        lrml = lrml.strip()
        lrml = lrml[lrml.find('if('):]
        lrml = lrml.replace('[', '(').replace(']', ')').replace(
            '{', '(').replace('}', ')')
        lrml = lrml.replace(').', ')')
        lrml = fix_then(lrml, prefix=prefix)

    # Fix errors is postprocessing
    if EXP_LOOP in simplifications:
        lrml = reverse_loop(lrml, prefix=prefix)
    if EXP_EXPRESSION in simplifications:
        lrml = reverse_resolve_expressions(
            lrml, fix_errors=True, prefix=prefix)
    if EXP_ATOM in simplifications:
        lrml = reverse_combine_rel_and_var(lrml, prefix=prefix)
    if EXP_DATA_AND_OR in simplifications:
        lrml = reverse_move_and_or_to_data_node(lrml)
    if EXP_UNIT in simplifications:
        lrml = reverse_units(lrml, prefix=prefix)

    # postprocessing
    if EXP_POSTPROCESS in simplifications:
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
    parser.add_argument('--seed', type=int, default=42,
                        help='The random seed.')
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
        if EXP_EXPRESSION in EXPERIMENTS[exp]:
            no_repeat_ngram_size = 9
            max_length = 512
        else:
            no_repeat_ngram_size = 13
            max_length = 1024

        #     Late import to be able to set CUDA_VISIBLE_DEVICES before torch is initialised
        from egice_training import transform_lists_to_ds, train, init_tokenizer

        model_name = 't5-amr' if 't5-amr' in args.model_path else args.model_path
        tokenizer_name = 't5-base' if 't5-amr' in args.model_path else args.model_path

        tokenizer = init_tokenizer(tokenizer_name)

        gen_kwargs0 = {
            "max_length": max_length,
            "num_beams": args.num_beams,
            "early_stopping": False,
            # "repetition_penalty": 1.2,
            # "no_repeat_ngram_size": no_repeat_ngram_size,
        }
        gen_kwargss = [gen_kwargs0]

        skipped = 0

        for split in [RANDOM_SPLIT, DOC_SPLIT]:
            for seed in [43, 44, 45]:
                if skipped < args.skip:
                    skipped += 1
                    print('Skipping', skipped)
                    continue

                exp_name = model_name + '_ir-rev_' + \
                    str(exp) + '_' + \
                    EXPERIMENT_NAMES[exp] + '_' + split + '_esp' + str(
                        args.early_stopping_patience) + '_e' + str(args.epochs) + '_' + str(seed)

                def post_process(x, y): return custom_postprocess_text(
                    x, y, simplifications=EXPERIMENTS[exp])

                ds = get_splits(tokenizer, df, split, EXPERIMENTS[exp])
                hyperparameters = dict(epochs=args.epochs, lr=lr, bs=bs, adafactor=False, weight_decay=1.000e-1, metric_for_best_model='lrml_f_score',
                                       num_warmup_steps=0, dataset=dataset_name, model_name=model_name, model_path=[args.model_path],
                                       post_process=post_process, lr_scheduler_type='linear', gradient_accumulation_steps=1,
                                       run_name=exp_name, eval_strategy=args.eval_strategy, eval_steps=args.steps,
                                       early_stopping_threshold=args.early_stopping_patience, max_length=max_length,
                                       seed=seed, write_predictions=args.write_predictions,
                                       evaluate_only=args.evaluate_only, evaluation_set=args.eval, runs=1, start_epoch=0, skip_evals=0,
                                       teacher_forcing_percentage=0.0, label_augmentation=0.0, retrain=True, mask_percentage=0.0, is_ir=False,
                                       fp16=False, split=split, experiment=EXPERIMENTS[exp], exp_num=exp)

                hyperparameters.update({k: v for k, v in gen_kwargs0.items()})

                print(hyperparameters)
                
                train(datasets=ds, hyperparameters=hyperparameters,
                      delete_model=True, gen_kwargss=gen_kwargss, post_process=post_process)
