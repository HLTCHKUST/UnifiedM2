from __future__ import absolute_import, division, print_function, unicode_literals

from utils.main_utils import *
from utils.const import misinfo_tasks
from utils.datasets import get_processor
import argparse
import glob
import logging
import json
import torch
import pandas as pd
from transformers import RobertaTokenizer

def get_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))

    parser.add_argument("--custom_exp_name", default="", type=str,
                        help="Custom information to add into the model save dir. use this when the param stays similar, but has implementation difference")

    parser.add_argument("--custom_eval_model_path", default="", type=str,
                        help="Directly supply a path of model to test. the way how i save a path differ time to time, so it's hard to have to keep track of all the syntax.")
    parser.add_argument("--log_path", default="result.log", type=str)

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model, results, tensorlogs, checkpoints will be written.")
    ## Other parameters
    parser.add_argument("--data_dir", default="/home/nayeon7lee", type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--experiment_name", default="", type=str,
                        help="Custom name of the experiment to run")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_zeroshot", action='store_true', help="Whether to test in zeroshot setting")
    parser.add_argument("--do_cross_val", action='store_true', help="Whether to do cross validation or not.")
    parser.add_argument("--cross_val_k", default=None, type=int, help="k value for cross validation")
    parser.add_argument("--cv_split_idx", default=None, type=int, help="split index for current iteration of cv")
    parser.add_argument("--test_event_name", default=None, type=str, help="event name for running PHEME dataset in event-based setup")


    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--batching", default='fit_main', type=str,
                        help="How to do batching from multiple data loaders of tasks")

    parser.add_argument("--patience", default=None, type=int,
                        help="patience for early stopping. if None, won't early stop")
    parser.add_argument("--eval_metric", default='f1', type=str,
                        help="Evaluation metric (loss, F1) to optimize for - early stopping")
    parser.add_argument("--remove_stopwords", action='store_true', help="Remove stopwords.")

    parser.add_argument('--use_combined_eval_metric', action='store_true', help="Decide whether to use all task's eval results or just the main results")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--optimizer", default='adamw', type=str,
                        help="choice of optimizer choice: [adamax, adamw]")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=20.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=float,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout", default=0, type=float)
    parser.add_argument("--loss_f", default='xentropy', type=str)
    parser.add_argument("--num_layers_to_freeze", default=0, type=int)
    parser.add_argument("--freeze_all", action='store_true', help="option of freezing the whole RoBERTa. simply use it as pretrained encoder")

    parser.add_argument('--dont_chunk', action='store_true',
                        help="Don't chunk aux dataset that is bigger than main task dataset size")
    parser.add_argument('--use_sampler4imbalance', action='store_true',
                        help="USe sampler to handle imbalanced dataset classes")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=1000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fewshot_train', type=int, default=None,
                        help="number of shot. provide -1 to do full shot (100 percentage of data)")
    parser.add_argument('--fewshot_train_ratio', type=float, default=None,
                        help="percentage of data to be used as trainset")
                        
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    args = parser.parse_args()

    return args

def main():
    args = get_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level=logging.WARN)
                        # level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

    logger.propagate = False
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # # Add additional Misinfo tasks
    for t_name in misinfo_tasks:
        processors[t_name] = get_processor(t_name)
        output_modes[t_name] = "classification"

    # Prepare all tasks
    args.task_name = args.task_name.lower().split(",")

    if len(args.task_name) > 1:
        print("MT!")
        args.mt_flag = True
        task_processors = []
        num_labels = []
        for task in args.task_name:
            if task not in processors:
                raise ValueError("Task not found: %s" % (task))
            processor = processors[task](args)
            task_processors.append(processor)
            num_labels.append(len(processor.get_labels()))
        args.output_mode = output_modes[task]  # all misinfo task is classification. so just use any task as index
    else:
        print("Single Task!")
        args.task_name = args.task_name[0]
        args.mt_flag = False
        if args.task_name not in processors:
            raise ValueError("Task not found: %s" % (args.task_name))

        processor = processors[args.task_name](args)

        args.output_mode = output_modes[args.task_name]
        label_list = processor.get_labels()
        num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels, 
                                            finetuning_task=args.task_name,
                                            loss_f=args.loss_f,
                                            custom_dropout=args.dropout,
                                            num_layers_to_freeze=args.num_layers_to_freeze, freeze_all=args.freeze_all)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # add parameter setting names to the
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    parameters_str = create_exp_name(args)
    args.output_dir = args.output_dir + parameters_str

    # Training
    if args.do_train:
        if args.mt_flag:
            train_datasets = [get_misinfo_datset(args, task, tokenizer, 'train')
                                for task in args.task_name]
            global_step, tr_loss = train_mt(args, train_datasets, model, tokenizer)
            logger.warning(" global_step = %s, average loss = %s", global_step, tr_loss)
        else:
            train_dataset = get_misinfo_datset(args, args.task_name, tokenizer, 'train')
            global_step, tr_loss = train(args, train_dataset, model, tokenizer)
            logger.warning(" global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_cross_val: # for ST only
        # train
        train_dataset = get_misinfo_datset(args, args.task_name, tokenizer, 'train')
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.warning(" global_step = %s, average loss = %s", global_step, tr_loss)

        # evaluate on test
        result, test_loss = evaluate(args, model, 'test', tokenizer)
        for key, value in sorted(result.items()):
            print("{}: {}".format(key, value))

        print("***** Save results on testset *****")
        result_path = "{}/cv/{}-test-results-{}.json".format(args.output_dir, args.task_name, args.cv_split_idx)
        with open(result_path, 'w') as f:
            json.dump(result, f)

    # Evaluation on TESTSET
    if args.do_test and args.local_rank in [-1, 0]:
        # Load a trained model and vocabulary that you have fine-tuned
        if args.custom_eval_model_path != "":
            best_model_path = "{}/{}".format(args.custom_eval_model_path, "best_model")
        else:
            best_model_path = "{}/{}".format(args.output_dir, "best_model")
        model = model_class.from_pretrained(best_model_path)
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
        model.to(args.device)

        logger.warning("Evaluate the model from : %s", best_model_path)

        print("***** Eval results on testset *****")

        if args.mt_flag:
            result, test_loss = evaluate_mt(args, model, 'test', tokenizer)

            result_objs = []
            for task in result.keys():
                print("______ {} ______".format(task))
                obj = { "task": task }

                for key in sorted(result[task].keys()):
                    # print("{}: {}".format(key, str(result[task][key])))
                    obj[key] = result[task][key]
                result_objs.append(obj)
            
        else:
            result, test_loss = evaluate(args, model, 'test', tokenizer)

            result_objs = { "task": args.task_name }

            with open("./{}".format(args.log_path), "a") as f:
                f.write("{} | {} | acc: {}, Macro-F1: {} | report: {} \n".format(args.task_name, args.custom_exp_name, result['acc'], result['macro_f1'], result['report']))
                
            for key, value in sorted(result.items()):
                print("{}: {}".format(key, value))
                result_objs[key]=value
            
        df = pd.DataFrame(result_objs)
        full_exp_name = "_".join(args.output_dir.split("/")[-2:])
        if not os.path.exists('./results/results_csv'):
            os.makedirs('./results/results_csv')
        df.to_csv ("./results/results_csv/{}.csv".format(full_exp_name), index = False, header=True)

        # print("***** Save results on testset *****")
        # result_path = "{}/test-results.json".format(args.output_dir)
        # with open(result_path, 'w') as f:
        #     json.dump(result, f)


    if args.do_zeroshot:
        # Load a trained model and vocabulary that you have fine-tuned
        if args.custom_eval_model_path != "":
            best_model_path = "{}/{}".format(args.custom_eval_model_path, "best_model")
        else:
            best_model_path = "{}/{}".format(args.output_dir, "best_model")
        model = model_class.from_pretrained(best_model_path)
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
        model.to(args.device)

        logger.warning("Evaluate the model from : %s", best_model_path)

        print("***** Eval results on testset zeroshot setting *****")
        if args.mt_flag:
            result, test_loss = evaluate_mt(args, model, 'test', tokenizer, main_task_only=True)
            for task in result.keys():
                print("______ {} ______".format(task))
                for key in sorted(result[task].keys()):
                    print("{}: {}".format(key, str(result[task][key])))
        else:
            result, test_loss = evaluate(args, model, 'test', tokenizer)
            for key, value in sorted(result.items()):
                print("{}: {}".format(key, value))

    


if __name__ == "__main__":
    main()