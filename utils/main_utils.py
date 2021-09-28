# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function


import json
import logging
import os
import random
import numpy as np
import pandas as pd
import itertools
from collections import defaultdict, Counter
import torch
from torch.utils.data.sampler import WeightedRandomSampler, BatchSampler
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, Dataset)
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adamax
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
from tqdm import tqdm, trange


from transformers import (WEIGHTS_NAME, BertConfig,
                            BertForSequenceClassification, BertTokenizer,
                            RobertaTokenizer,
                            XLMConfig, XLMForSequenceClassification,
                            XLMTokenizer, XLNetConfig,
                            XLNetForSequenceClassification,
                            XLNetTokenizer,
                            DistilBertConfig,
                            DistilBertForSequenceClassification,
                            DistilBertTokenizer)

from transformers import AdamW, WarmupLinearSchedule
from model.roberta_mtl import RobertaForMTL
from model.roberta_single import RobertaForSequenceClassification, RobertaConfig
from utils.datasets import *
from utils.metric import compute_metrics, get_wrong_results, get_FPN_TPN, get_statistics_about_correct_and_wrong, evaluate_true_results, evaluate_false_results
from utils.sampler import ImbalancedDatasetSampler

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig,
                                                                                RobertaConfig, DistilBertConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'roberta_mt': (RobertaConfig, RobertaForMTL, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer)
}


def save_best_eval(args, model, tokenizer, results, epoch=None):
    # save trained model
    output_dir = os.path.join(args.output_dir, 'best_model')
    save_pretrained_model(output_dir, model, tokenizer, args)

    # add details about the experiment
    results['task'] = args.task_name
    results['learning_rate'] = args.learning_rate
    results['optimizer'] = args.optimizer
    results['warmup_steps'] = args.warmup_steps
    results['batching'] = args.batching
    results['per_gpu_train_batch_size'] = args.per_gpu_train_batch_size
    results['eval_metric'] = args.eval_metric
    if epoch != None:
        results['epoch'] = epoch

    # save the results
    with open("{}/best_val_result_w_metric_{}.json".format(args.output_dir, args.eval_metric), 'w') as f:
        json.dump(results, f)

def create_exp_name(args):

    default_name = "lr.{}_bz.{}".format(args.learning_rate,args.per_gpu_train_batch_size)
    additional_name = ""

    if args.custom_exp_name != "":
        additional_name = additional_name + "_" + args.custom_exp_name
    return default_name + '_' + additional_name

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def save_pretrained_model(model_save_path, model, tokenizer, args):
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    print("Saving new best model to {}".format(model_save_path))
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(model_save_path, 'training_args.bin'))


def train_mt(args, train_datasets, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter('{}/tensorboard'.format(args.output_dir))

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    sampler_fn = ImbalancedDatasetSampler if args.use_sampler4imbalance else RandomSampler
    train_dls = [DataLoader(dataset, batch_size=args.train_batch_size, sampler=sampler_fn(dataset))
            for dataset in train_datasets]
    dl_len_max = max([len(dl) for dl in train_dls])

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (dl_len_max // args.gradient_accumulation_steps) + 1
    else:
        t_total = dl_len_max // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    if args.optimizer == 'adamw':
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    else:
        optimizer = Adamax(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps*t_total, t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_datasets))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    main_task_name = args.task_name[0]
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    val_min_loss, patience_cnt, val_max_f1 = float('inf'), 0, -1

    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for e in train_iterator:
        epoch_tr_loss = 0.0
        epoch_iterator = tqdm(itertools.zip_longest(*train_dls), desc="Iteration {}".format(e), total=dl_len_max)
        model.train()
        for step, batches in enumerate(epoch_iterator):

            # stop iterating when the main task batch is exhausted. By doing this, we are theoretically "chunking" dataset that are "larger" than main's dataset
            if batches[0] is None and not args.dont_chunk:
                epoch_iterator.close()
                train_iterator.close()
                break

            mt_loss = torch.zeros(1).cuda()
            for t_id, batch in enumerate(batches):
                if batch is None:
                    continue
                # batch: list of batches for each data
                
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'labels':         batch[3],
                          'task':           batch[4],
                          'guids':          batch[5]
                          }
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = model(**inputs)
                loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
                
                if args.n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                # joint loss
                mt_loss += loss

            if args.fp16:
                with amp.scale_loss(mt_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                mt_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += mt_loss.item()
            epoch_tr_loss += mt_loss.item()

            log = "(E{}, T{}) TRAIN avg loss: {:.3f}".format(e, t_id, epoch_tr_loss / (step+1))
            epoch_iterator.set_description(log)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    save_pretrained_model(output_dir, model, tokenizer, args)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.patience is not None and args.evaluate_during_training:
            if args.use_combined_eval_metric:
                results, val_loss = evaluate_mt(args, model, 'dev', tokenizer, main_task_only=False)
                val_f1 = sum([results[key]['macro_f1'] for key in results])
                val_loss = sum(val_loss)
            else:
                results, val_loss = evaluate_mt(args, model, 'dev', tokenizer, main_task_only=True)
                val_f1 = results[main_task_name]['macro_f1']
                val_loss = val_loss[0]

            if args.eval_metric == 'loss':
                if val_loss < val_min_loss:
                    val_min_loss = val_loss
                    patience_cnt = 0 # reset patience cnt
                    results['val_loss'] = val_loss

                    save_best_eval(args, model, tokenizer, results, e)

                else:
                    patience_cnt += 1
            elif args.eval_metric == 'f1':
                if val_f1 > val_max_f1:
                    logger.info("Improve in macro_f1")
                    val_max_f1 = val_f1
                    patience_cnt = 0 # reset patience cnt
                    results['val_loss'] = val_loss

                    save_best_eval(args, model, tokenizer, results, e)

                else:
                    patience_cnt += 1
                    logger.info("NO improvement in macro_f1", patience_cnt)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step

"""
If main_task_only==True: Returns results dict and loss arr for ONLY main task,
Else, Returns results dict and loss arr (loss in the order of tasks) for ALL tasks
"""
def evaluate_mt(args, model, phase, tokenizer, main_task_only=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    results = defaultdict(dict)
    eval_losses = []

    tasks_to_evaluate = [args.task_name[0]] if main_task_only else args.task_name

    for eval_task in tasks_to_evaluate:
        if phase == 'dev':
            eval_dataset = get_misinfo_datset(args, eval_task, tokenizer, 'dev')
        elif phase == 'test':
            eval_dataset = get_misinfo_datset(args, eval_task, tokenizer, 'test')

        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        logger.info("***** Running evaluation {} *****".format(eval_task))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        eval_pbar = tqdm(eval_dataloader, desc="MT Evaluating")
        for batch in eval_pbar:
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'labels':         batch[3],
                          'task':           batch[4],
                          'guids':          batch[5]
                          }
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            log = "EVAL avg loss: {:.3f}".format(eval_loss / nb_eval_steps)
            eval_pbar.set_description(log)

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)

        result = compute_metrics(eval_task, preds, out_label_ids)
        results[eval_task].update(result)
        eval_losses.append(eval_loss)

    return results, eval_losses

def train(args, train_dataset, model, tokenizer):
    """ Train the ST model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter('{}/tensorboard'.format(args.output_dir))

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = ImbalancedDatasetSampler(train_dataset) if args.use_sampler4imbalance else RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    if args.optimizer == 'adamw':
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    elif args.optimizer == 'adamax':
        optimizer = Adamax(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    else:
        print("Wrong optimizer name given!")
        exit(1)

    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps*t_total, t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    val_min_loss, patience_cnt, val_max_f1 = float('inf'), 0, -1

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for e in train_iterator:
        epoch_tr_loss = 0.0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration under Epoch {}".format(e), disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'labels':         batch[3],
                      'guids':          batch[5]
                      }
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            epoch_tr_loss += loss.item()

            log = "(E{}) TRAIN avg loss: {:.3f}".format(e, epoch_tr_loss / (step + 1))
            epoch_iterator.set_description(log)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.patience is not None and args.evaluate_during_training:
            results, val_loss = evaluate(args, model, 'dev', tokenizer)
            if args.eval_metric == 'loss':
                if val_loss < val_min_loss:
                    val_min_loss = val_loss
                    patience_cnt = 0 # reset patience cnt
                    results['loss'] = val_min_loss
                    save_best_eval(args, model, tokenizer, results, e)

                else:
                    patience_cnt += 1
            elif args.eval_metric == 'f1':
                if results['macro_f1'] > val_max_f1:
                    logger.info("Improve in macro_f1")
                    val_max_f1 = results['macro_f1']
                    patience_cnt = 0 # reset patience cnt
                    save_best_eval(args, model, tokenizer, results, e)

                else:
                    patience_cnt += 1

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, phase, tokenizer):
    """
        Evaluate ST model
        If phase == dev, only return the results.
        If phase == test, return the results, and SAVE the results
    """
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        if args.task_name == 'sst-2':
            eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)
        else:
            if phase == 'dev':
                eval_dataset = get_misinfo_datset(args, eval_task, tokenizer, 'dev')
            elif phase == 'test':
                eval_dataset = get_misinfo_datset(args, eval_task, tokenizer, 'test')

            if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
                os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        eval_pbar = tqdm(eval_dataloader, desc="Evaluating")
        for batch in eval_pbar:
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'labels':         batch[3],
                          'guids':          batch[5]
                          }
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1
            log = "EVAL avg loss: {:.3f}".format(eval_loss / nb_eval_steps)
            eval_pbar.set_description(log)

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        logger.warning("Final Avg Eval Loss %s", eval_loss)

        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)

        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

    return results, eval_loss

gradient = None # gradient is global. For caching the intermediate gradient value
def hook_fun(grad):
    global gradient
    gradient = grad