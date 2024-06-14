# Follows the translation setup https://github.com/huggingface/notebooks/blob/main/examples/translation.ipynb
import math
import os
import random
import shutil

import evaluate
import numpy as np
import wandb
from torch import LongTensor
from torch.utils.data import DataLoader
from training_utils import *
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, get_scheduler)
from dotenv import load_dotenv


load_dotenv()

tokenizer = None

REPLACEMENT = None
REPLACED_PREFIX = None
AUG_DS_NUMBER = 0

metric = evaluate.load('bleu')


def init_tokenizer(model_name):
    global tokenizer, REPLACEMENT, REPLACED_PREFIX
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.sep_token is None:
        tokenizer.add_tokens(
            ['<sep>', '<fun>', '<con>', '<deon>', '<mask>'], special_tokens=True)
        tokenizer.add_tokens(['<sep>', '<mask>'], special_tokens=True)
        tokenizer.sep_token = '<sep>'
        tokenizer.sep_token_id = tokenizer.convert_tokens_to_ids(
            tokenizer.sep_token)
        tokenizer.mask_token = '<mask>'
        tokenizer.mask_token_id = tokenizer.convert_tokens_to_ids(
            tokenizer.mask_token)

    REPLACED_PREFIX = tokenizer('Translate English to')['input_ids'][:-1]
    REPLACEMENT = LongTensor(tokenizer('Fix')['input_ids'][:-1])
    return tokenizer


# Includes source code to train to models in parallel rather than iteratively
def train(datasets, hyperparameters, delete_model=True, gen_kwargss=None, post_process=None):
    train_predictions = None
    # get project from .env
    with wandb.init(project=os.environ['WANDB_PROJECT'], entity=os.environ['WANDB_ENTITY'], config=hyperparameters):
        if hyperparameters:
            wandb.log(hyperparameters)
        config = wandb.config
        wandb.run.name = config['run_name']

        set_seed(config.seed)

        models = [AutoModelForSeq2SeqLM.from_pretrained(
            i).cuda() for i in config.model_path]

        data_collator = DataCollatorForSeq2Seq(
            tokenizer, model=models[0])  # , pad_to_multiple_of=8)
        train_dataloader = DataLoader(
            datasets['train'], shuffle=True, collate_fn=data_collator, batch_size=config.bs
        )
        eval_dataloader = DataLoader(
            datasets['valid'], collate_fn=data_collator, batch_size=config.bs)
        test_dataloader = DataLoader(
            datasets['test'], collate_fn=data_collator, batch_size=config.bs)

        # Optional Dataloaders
        all_dataloader = None
        test_oracle_dataloader = None
        test_no_sep_dataloader = None
        if 'test_oracle' in datasets.keys() and datasets['test_oracle'] is not None:
            test_oracle_dataloader = DataLoader(
                datasets['test_oracle'], collate_fn=data_collator, batch_size=config.bs)
        if 'test_no_sep' in datasets.keys() and datasets['test_no_sep'] is not None:
            test_no_sep_dataloader = DataLoader(
                datasets['test_no_sep'], collate_fn=data_collator, batch_size=config.bs)
        if 'all' in datasets.keys() and datasets['all'] is not None:
            all_dataloader = DataLoader(
                datasets['all'], collate_fn=data_collator, batch_size=config.bs)

        optimizers = [get_optimizer(
            model, lr=config.lr, weight_decay=config.weight_decay) for model in models]
        scaler = torch.cuda.amp.GradScaler()

        dl = train_dataloader

        num_update_steps_per_epoch = math.ceil(
            len(dl) / config.gradient_accumulation_steps)
        if len(models) == 1:
            train_runs = [config.runs]
        elif len(models) == 2:
            train_runs = [1 if config.retrain else 0, config.runs - 1]
        else:
            raise Exception('Not implemented')
        max_train_steps_per_run = config.epochs * num_update_steps_per_epoch
        max_train_steps = config.epochs * \
            num_update_steps_per_epoch * sum(train_runs)

        lr_schedulers = [get_scheduler(
            name=config.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=config.num_warmup_steps,
            num_training_steps=max_train_steps_per_run * train_runs[i],
        ) for i, optimizer in enumerate(optimizers)]

        total_batch_size = config.bs * config.gradient_accumulation_steps

        progress_bar = tqdm(range(max_train_steps))
        completed_steps = 0

        def log_once(text):
            if completed_steps < (config.runs if config.retrain else config.runs-1):
                print(text)

        starting_epoch = 0

        current_best = {config.metric_for_best_model: 0.}
        last_improvement = 0

        losses = []

        for epoch in range(starting_epoch, config.epochs):
            if last_improvement >= config.early_stopping_threshold:
                break
            losses = []
            [model.train() for model in models]
            for step, batch in enumerate(dl):
                if config.retrain:
                    log_once('Retraining')
                    with torch.cuda.amp.autocast(enabled=config.fp16):
                        outputs = models[0](batch["input_ids"].cuda(),
                                            attention_mask=batch["attention_mask"].cuda(
                        ),
                            labels=batch["labels"].cuda(
                        ) if not config.is_ir else batch["ir_labels0"].cuda(),
                            decoder_input_ids=batch["decoder_input_ids"].cuda() if not config.is_ir else batch["ir_0decoder_input_ids"].cuda())
                        # labels=batch["ir_labels"].cuda(),
                        # decoder_input_ids=batch["ir_decoder_input_ids"].cuda())
                    loss = outputs.loss
                    loss = loss / config.gradient_accumulation_steps
                    log_once('Loss model 1: ' + str(loss))
                    scaler.scale(loss).backward()
                    losses.append(loss.detach().cpu())

                    # if step % conf.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    scaler.step(optimizers[0])
                    scaler.update()
                    lr_schedulers[0].step()
                    optimizers[0].zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1

                if epoch >= config.start_epoch:
                    for i in range(config.runs-1):
                        log_once('Training second model')
                        random_number = random.random()
                        if random_number < config.label_augmentation and not config.is_ir:
                            log_once('Label augmentation')
                            batch = generate_new_inputs(batch, batch['ir_' + str(
                                epoch % AUG_DS_NUMBER) + 'decoder_input_ids'], i + 1, mask_percentage=config.mask_percentage)
                        elif random_number < config.label_augmentation + config.teacher_forcing_percentage:
                            if not config.retrain:
                                log_once('Rerun model 1 for teacher forcing')
                                with torch.no_grad():
                                    with torch.cuda.amp.autocast(enabled=config.fp16):
                                        outputs = models[0](batch["input_ids"].cuda(),
                                                            attention_mask=batch["attention_mask"].cuda(
                                        ),
                                            labels=batch["labels"].cuda(
                                        ),
                                            decoder_input_ids=batch["decoder_input_ids"].cuda())
                            log_once('Teacher forcing')
                            batch = generate_new_inputs(batch, torch.argmax(
                                outputs.logits, axis=-1), i + 1, mask_percentage=config.mask_percentage)
                        else:
                            log_once('Generation from model 1')
                            with torch.no_grad():
                                with torch.cuda.amp.autocast(enabled=config.fp16):
                                    generated_tokens = models[0].generate(
                                        batch["input_ids"].cuda(),
                                        attention_mask=batch["attention_mask"].cuda(
                                        ),
                                        **gen_kwargss[0]
                                    ).cpu()
                            batch = generate_new_inputs(
                                batch, generated_tokens, i + 1, mask_percentage=config.mask_percentage)
                        with torch.cuda.amp.autocast(enabled=config.fp16):
                            outputs = models[-1](batch["input_ids"].cuda(),
                                                 attention_mask=batch["attention_mask"].cuda(
                            ),
                                labels=batch["labels"].cuda(),
                                decoder_input_ids=batch["decoder_input_ids"].cuda())
                        loss = outputs.loss
                        loss = loss / config.gradient_accumulation_steps
                        log_once('Loss model 2: ' + str(loss))
                        scaler.scale(loss).backward()
                        losses.append(loss.detach().cpu())

                    if config.runs > 1:
                        log_once('Optim model 2')
                        scaler.step(optimizers[-1])
                        scaler.update()
                        lr_schedulers[-1].step()
                        optimizers[-1].zero_grad()
                        progress_bar.update(1)
                        completed_steps += 1

                if completed_steps >= max_train_steps:
                    break

            if epoch >= config.skip_evals:
                # if epoch % 1 == 0:
                # train_loss = torch.stack(losses).mean().item()
                current_metric = evaluate_lrml_mtd(config, tokenizer, models, eval_dataloader, gen_kwargss,
                                                   config.fp16, epoch, config.runs, config.start_epoch, custom_postprocess=post_process, calc_loss=config.get('calc_loss', False))
                # if train_predictions is None and all_dataloader is not None and config.get('calc_loss', False) and train_loss < current_metric['eval_loss']:
                if all_dataloader is not None and config.get('calc_loss', False) and epoch == config.get('pred_epoch', 3):
                    [model.save_pretrained(
                        'models/' + wandb.run.name + '_pred') for i, model in enumerate(models)]

                wandb.log(current_metric)
                if current_metric[config.metric_for_best_model] > current_best[config.metric_for_best_model]:
                    current_best = current_metric
                    [model.save_pretrained(
                        'models/' + wandb.run.name + '_' + str(i)) for i, model in enumerate(models)]
                    last_improvement = 0
                else:
                    last_improvement += 1

        # Report best validation score
        wandb.log(current_best)

        print('TEST SET: ')
        model = AutoModelForSeq2SeqLM.from_pretrained(
            'models/' + wandb.run.name + '_0').cuda()
        current_metric = evaluate_lrml_mtd(config, tokenizer, [model], test_dataloader, gen_kwargss,
                                           config.fp16, epoch, config.runs, config.start_epoch, custom_postprocess=post_process)
        wandb.log({k + '_test': v for k, v in current_metric.items()})

        if test_oracle_dataloader is not None:
            print('ORACLE TEST SET: ')
            current_metric = evaluate_lrml_mtd(config, tokenizer, [model], test_oracle_dataloader, gen_kwargss,
                                               config.fp16, epoch, config.runs, config.start_epoch, custom_postprocess=post_process)
            wandb.log({k + '_test_oracle': v for k,
                      v in current_metric.items()})

        if test_no_sep_dataloader is not None:
            print('ORACLE TEST SET: ')
            current_metric = evaluate_lrml_mtd(config, tokenizer, [model], test_no_sep_dataloader, gen_kwargss,
                                               config.fp16, epoch, config.runs, config.start_epoch, custom_postprocess=post_process)
            wandb.log({k + '_test_no_sep': v for k,
                      v in current_metric.items()})

        all_predictions = None
        if all_dataloader is not None:
            all_predictions = get_predictions(
                tokenizer, model, all_dataloader, gen_kwargss[0])
            if config.get('calc_loss', False):
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    'models/' + wandb.run.name + '_pred').cuda()
                print('Calculating train predictions')
                train_predictions = get_predictions(
                    tokenizer, model, all_dataloader, gen_kwargss[0])
                wandb.log({'prediction_epoch': config.get('pred_epoch', 3)})
                shutil.rmtree('models/' + wandb.run.name + '_pred')

        if delete_model:
            shutil.rmtree('models/' + wandb.run.name + '_0')

        return 'models/' + wandb.run.name + '_0', all_predictions, train_predictions


def evaluate_lrml_mtd(conf, tokenizer, models, data_loader, gen_kwargss, fp16, epoch, runs, start_epoch, custom_postprocess=None, predictions_path=None, calc_loss=False):
    [model.eval() for model in models]
    generated_list = []
    ir_list = []
    label_list = []
    input_list = []
    losses = []
    eval_progress_bar = tqdm(range(len(data_loader)), leave=False)
    rand_batch = random.randint(0, len(data_loader)-1)
    rand_sample = random.randint(0, conf.bs-1)
    for step, batch in enumerate(data_loader):
        runs = 1 if epoch < start_epoch else runs
        for i in range(runs):
            if i > 0:
                if i == 1:
                    ir_list.append(generated_tokens)
                batch = generate_new_inputs(
                    batch, generated_tokens, i, mask_percentage=0.0)
                if step == rand_batch:
                    print(tokenizer.decode(batch['input_ids'][min(batch['input_ids'].shape[0] - 1, rand_sample)].tolist(
                    ), skip_special_tokens=False).replace('<sep> ', '\n').split('<pad>')[0])
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=fp16):
                    generated_tokens = models[min(i, len(models)-1)].generate(
                        batch["input_ids"].cuda(),
                        attention_mask=batch["attention_mask"].cuda(),
                        **gen_kwargss[min(i, 1)]
                    ).cpu()

            # if calc_loss:
            #     with torch.no_grad():
            #         with torch.cuda.amp.autocast(enabled=fp16):
            #             outputs = models[min(i, len(models)-1)](batch["input_ids"].cuda(),
            #                                                     attention_mask=batch["attention_mask"].cuda(
            #             ),
            #                 labels=batch["labels"].cuda(
            #             ),
            #                 decoder_input_ids=batch["decoder_input_ids"].cuda())

            #     losses.append(outputs.loss.detach().cpu())

        if step == rand_batch:
            # decode without special tokens
            print(tokenizer.decode(generated_tokens[min(
                generated_tokens.shape[0] - 1, rand_sample)].tolist(), skip_special_tokens=True))
            print_labels = batch["labels"][min(
                batch["labels"].shape[0] - 1, rand_sample)].cpu().tolist()
            # replace -100 in the labels as we can't decode them.
            print_labels = [
                x if x != -100 else tokenizer.pad_token_id for x in print_labels]
            print(tokenizer.decode(print_labels, skip_special_tokens=True))
            print()

        if runs == 1 and conf.is_ir:
            labels = batch["ir_labels0"].cpu()
        else:
            labels = batch["labels"].cpu()
        input_ids = batch["input_ids"].cpu()

        generated_list.append(generated_tokens)
        label_list.append(labels)
        input_list.append(input_ids)
        eval_progress_bar.update(1)

    if not ir_list:
        ir_list = generated_list

    eval_metric = compute_metrics(
        (padded_concat(tokenizer, generated_list), padded_concat(tokenizer, label_list), padded_concat(tokenizer, input_list), padded_concat(tokenizer, ir_list)), tokenizer, custom_postprocess)
    if predictions_path:
        write_predictions(path=predictions_path, tokenizer=tokenizer, inputs=padded_concat(tokenizer, input_list), label_ids=padded_concat(
            tokenizer, label_list), predictions=padded_concat(tokenizer, generated_list), metrics=eval_metric)
    eval_progress_bar.close()
    eval_metric['eval_epoch'] = epoch
    print(eval_metric)
    # if calc_loss:
    #     eval_metric['eval_loss'] = torch.stack(losses).mean().item()
    return eval_metric


def generate_new_inputs(input_dict, output_dict, run_number, remove_input=False, remove_output=False, mask_percentage=0.1):
    """
    New function to concatenate the output to the input during the training
    Args:
        input_dict (`Dict[str, Union[torch.Tensor, Any]]`):
            The inputs and targets of the model.
            The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
            argument `labels`. Check your model's documentation for all accepted arguments.
        output_dict:
        run_number: Retrive prefix tokens dependent to decoding run_number.
    Return:
        `Dict[str, Union[torch.Tensor, Any]]`: The tensor with training loss on this batch.
    """
    bs = input_dict['input_ids'].shape[0]
    new_inputs = []
    for i in range(bs):
        t1 = input_dict['input_ids'][i].cpu()
        # Whichever token comes first, EOS or SEP
        t1_s = t1[:torch.where((t1 == tokenizer.eos_token_id) | (
            t1 == tokenizer.sep_token_id))[0][0] + 1]
        t1_s = t1_s[torch.where((t1_s != tokenizer.pad_token_id))]
        if run_number == 1:
            # Adjust the prefix to fix LegalRuleML by removing 'translate English' and replacing 'to' with 'fix'
            t1_s = torch.cat((REPLACEMENT, t1_s[len(REPLACED_PREFIX):]))
            if remove_input:
                # Find : and remove everything after it
                t1_s = t1_s[:torch.where((t1 == 10))[0][0] + 1]
            else:
                # Use sep token in the end
                t1_s[-1] = tokenizer.sep_token_id
        if output_dict is not None:
            o1 = output_dict[i].cpu()
            o1_s = o1[torch.where((o1 != tokenizer.pad_token_id) & (
                o1 != tokenizer.bos_token_id))]

        cat1 = torch.cat((t1_s, o1_s))
        if remove_output:
            cat1 = t1_s
        # Mask mask_percentage % of the tokens
        keep = torch.empty_like(cat1).bernoulli_(1 - mask_percentage).bool()
        torch.where(keep, cat1, torch.empty_like(
            cat1).fill_(tokenizer.mask_token_id))
        new_inputs.append(cat1)

    input_dict.update(tokenizer.pad(
        {'input_ids': new_inputs}, return_tensors='pt'))
    # Move to GPU
    input_dict = {k: v.cuda() for k, v in input_dict.items()}
    return input_dict


# https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def get_predictions(tokenizer, model, data_loader, gen_kwargs):
    model.eval()
    generated_list = []
    eval_progress_bar = tqdm(range(len(data_loader)), leave=False)
    for batch in data_loader:
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                generated_tokens = model.generate(
                    batch["input_ids"].cuda(),
                    attention_mask=batch["attention_mask"].cuda(),
                    **gen_kwargs
                ).cpu()

        generated_list.append(generated_tokens)
        eval_progress_bar.update(1)

    decoded_preds = tokenizer.batch_decode(padded_concat(
        tokenizer, generated_list), skip_special_tokens=True)
    eval_progress_bar.close()
    return decoded_preds
