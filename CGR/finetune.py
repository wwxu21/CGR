import faiss
import argparse
import glob
import logging
import os
import random
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
sys.path.append("./")
import copy
from CGR.utils import convert_examples_to_features, processors
from CGR.retriever_utils import read_amr, read_fact
import torch
import numpy as np
from torch.utils.data import TensorDataset
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup, AutoConfig, AutoTokenizer
from CGR.retriever import Retriever
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
from transformers import WEIGHTS_NAME, RobertaConfig, RobertaForMultipleChoice, RobertaTokenizer

config_class_reader, model_class_reader, tokenizer_class_reader = RobertaConfig, RobertaForMultipleChoice, RobertaTokenizer
config_class_retriever, model_class_retriever, tokenizer_class_retriever = AutoConfig, Retriever, AutoTokenizer

def feature2list(InputFeature):
    feature_list = []
    for feature_x in InputFeature.choices_features:
        input_ids, input_mask, segment_ids, amr_ids, facts_adj = feature_x['input_ids'], feature_x['input_mask'], feature_x[
            'segment_ids'], feature_x['amr_ids'], feature_x['facts_adj']
        feature_list.append((input_ids, input_mask, segment_ids, amr_ids, facts_adj))
    return feature_list

def select_field(features, field):
    return [[choice[field] for choice in feature.choices_features] for feature in features]

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def train(args, reader, retriever, reader_tokenizer, retriever_tokenizer, fact_data, AMRDataLoader=None):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    AMR_dataloader = AMRDataLoader(args, fact_data, reader_tokenizer, retriever_tokenizer, evaluate=False, test=False)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(AMR_dataloader.qa_f['train']) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(AMR_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    args.save_steps = t_total // args.save_steps
    args.logging_steps = t_total // args.logging_steps

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in reader.named_parameters() if
                       not any(nd in n for nd in no_decay)] + [p for n, p in retriever.named_parameters() if
                       not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate,
        },
        {"params": [p for n, p in reader.named_parameters() if
                    any(nd in n for nd in no_decay) ] + [p for n, p in retriever.named_parameters() if
                    any(nd in n for nd in no_decay) ], "weight_decay": 0.0,
         "lr": args.learning_rate, },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_ratio * t_total, num_training_steps=t_total
    )
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        reader, optimizer = amp.initialize(reader, optimizer, opt_level=args.fp16_opt_level)
        retriever = amp.initialize(retriever, opt_level=args.fp16_opt_level)
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1 and not isinstance(reader, torch.nn.DataParallel):
        reader = torch.nn.DataParallel(reader)
        retriever = torch.nn.DataParallel(retriever)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        reader = torch.nn.parallel.DistributedDataParallel(
            reader, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )
        retriever = torch.nn.parallel.DistributedDataParallel(
            retriever, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(AMR_dataloader.qa_f['train']))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_dev_acc = 0.0
    best_steps = 0

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility
    for i_epoch, _ in enumerate(train_iterator):
        if i_epoch > 0:
            del batch, loss
            torch.cuda.empty_cache()
        if i_epoch < 4:
            AMR_dataloader.retrieve_fact(retriever)
            retriever_save = model_class_retriever.from_pretrained(
                args.retriever_name,
                cache_dir="./cache_file/",
                from_tf=bool(".ckpt" in args.retriever_name), )
            state_dict_save = retriever.state_dict().copy()
            retriever_save.load_state_dict(state_dict_save)
        epoch_iterator = tqdm(AMR_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        # results = evaluate(args, reader, retriever, reader_tokenizer, retriever_tokenizer, fact_data)
        # results_test = evaluate(args, reader, retriever, reader_tokenizer, retriever_tokenizer, fact_data,
        #                         test=True)
        for step, batch in enumerate(epoch_iterator):

            reader.train()
            retriever.train()
            # with torch.no_grad():
            batch_reader, batch_retriever = batch[:4], batch[4:]
            batch_retriever = tuple(t.to(args.device) if torch.is_tensor(t) else t for t in batch_retriever)
            inputs_retriever = {
                "query_ids": batch_retriever[0],
                "query_mask": batch_retriever[1],
                "fact_vec": batch_retriever[2],
                "iter_mask": batch_retriever[3],
                "label":batch_reader[3],
            }

            retriever_outputs = retriever(**inputs_retriever)
            loss_retriever = retriever_outputs[0]

            batch_reader = tuple(t.to(args.device) for t in batch_reader)
            inputs_reader = {
                "input_ids": batch_reader[0],
                "attention_mask": batch_reader[1],
                "token_type_ids": None,
                "labels": batch_reader[3],
            }
            reader_outputs = reader(**inputs_reader)
            loss_reader = reader_outputs[0]  #  reader_outputs are always tuple in transformers (see doc)
            inputs_RL = {
                "policy": retriever_outputs[1],
                "pred": reader_outputs[1],
                "label": batch_reader[3],
                "iter_segmentation": batch_retriever[4]
            }
            loss_RL = retriever.RLlearning(**inputs_RL)
            loss = loss_reader + loss_retriever + loss_RL
            # loss = retriever_loss
            # loss_path = reader_outputs[2]
            # loss = loss + args.path_weight * loss_path
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(reader.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                reader.zero_grad()
                retriever.zero_grad()
                global_step += 1
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics

                    if (
                            args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well

                        results = evaluate(args, reader, retriever, reader_tokenizer, retriever_tokenizer, fact_data, AMRDataLoader=AMRDataLoader)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                        if results["eval_acc"] > best_dev_acc:
                            best_dev_acc = results["eval_acc"]
                            best_steps = global_step
                        if args.do_test:
                            results_test = evaluate(args, reader, retriever, reader_tokenizer, retriever_tokenizer, fact_data, test=True, AMRDataLoader=AMRDataLoader)
                            for key, value in results_test.items():
                                tb_writer.add_scalar("test_{}".format(key), value, global_step)
                            logger.info(
                                "test acc: %s, loss: %s, global steps: %s",
                                str(results_test["eval_acc"]),
                                str(results_test["eval_loss"]),
                                str(global_step),
                            )
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logger.info(
                        "Average loss: %s at global step: %s",
                        str((tr_loss - logging_loss) / args.logging_steps),
                        str(global_step),
                    )
                    logging_loss = tr_loss

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step, best_steps


def evaluate(args, reader, retriever, reader_tokenizer, retriever_tokenizer, fact_data, prefix="", test=False, AMRDataLoader=None):
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

        AMR_dataloader = AMRDataLoader(args, fact_data, reader_tokenizer, retriever_tokenizer, evaluate=not test, test=test)
        # multi-gpu evaluate
        if test:
            dataID = "test"
        else:
            dataID = 'dev'
        if args.n_gpu > 1 and not isinstance(reader, torch.nn.DataParallel):
            reader = torch.nn.DataParallel(reader)
            retriever = torch.nn.DataParallel(retriever)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(AMR_dataloader.qa_f[dataID]))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        AMR_dataloader.retrieve_fact(retriever)
        for batch in tqdm(AMR_dataloader, desc="Evaluating"):
            reader.eval()
            batch_qa, batch_retriever = batch[:4], batch[4:]
            batch_qa = tuple(t.to(args.device) for t in batch_qa)

            with torch.no_grad():

                inputs = {
                    "input_ids": batch_qa[0],
                    "attention_mask": batch_qa[1],
                    "token_type_ids": None,
                    "labels": batch_qa[3],
                }
                outputs = reader(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=1)
        acc = simple_accuracy(preds, out_label_ids)
        result = {"eval_acc": acc, "eval_loss": eval_loss}
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, "is_test_" + str(test).lower() + "_eval_results.txt")

        with open(output_eval_file, "a") as writer:
            logger.info("***** Eval results {} *****".format(str(prefix) + " is test:" + str(test)))
            writer.write("reader           =%s\n" % str(prefix))
            writer.write(
                "total batch size=%d\n"
                % (
                        args.per_gpu_train_batch_size
                        * args.gradient_accumulation_steps
                        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)
                )
            )
            writer.write("train num epochs=%d\n" % args.num_train_epochs)
            writer.write("fp16            =%s\n" % args.fp16)
            writer.write("max seq length  =%d\n" % args.max_len)
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False, test=False):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    processor = processors[task]()

    # Load data features from cache or dataset file
    if evaluate:
        cached_mode = "dev"
    elif test:
        cached_mode = "test"
    else:
        cached_mode = "train"
    assert not (evaluate and test)
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            cached_mode,
            list(filter(None, args.model_name.split("/"))).pop(),
            str(args.max_len),
            str(task),
        ),
    )

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        # features2 = torch.load(cached_features_file[:-4])
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if evaluate:
            examples = processor.get_dev_examples(args.data_dir)
        elif test:
            examples = processor.get_test_examples(args.data_dir)
        else:
            examples = processor.get_train_examples(args.data_dir)
        logger.info("Training number: %s", str(len(examples)))

        features = convert_examples_to_features(
            examples,
            label_list,
            args.max_len,
            tokenizer,
            sep_token_extra=False,
            pad_on_left=False,
            pad_token_segment_id=0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset

    all_input_ids = torch.tensor(select_field(features, "input_ids"), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(features, "input_mask"), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(features, "segment_ids"), dtype=torch.long)
    all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_name",
        default='race',
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--model_name",
        default='roberta-base',
        type=str,
        required=False,
        help="Path to pre-trained model",
    )
    parser.add_argument(
        "--retriever_name",
        default="sentence-transformers/bert-base-nli-cls-token",
        type=str,
        required=False,
        help="Path to retreiver pre-trained model",
    )
    parser.add_argument(
        "--post",
        default='',
        type=str,
        required=False,
        help="Path to data",
    )
    # Other parameters
    parser.add_argument(
        "--cache_dir",
        default="./cache_file/",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_len",
        default=256,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run test on the test set")
    parser.add_argument(
        "--evaluate_during_training", default=True, help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", default=True, help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--max_fact", default=15, type=int, help="max number of fact as context in reader")
    parser.add_argument("--hop", default=2, type=int, help="number of multi-hop inference")
    parser.add_argument("--kfact", default=10, type=int, help="number of top k facts.")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--drop", default=0.1, type=float, help="dropout rate for hidden state.")
    parser.add_argument("--weight_decay", default=0.1, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=4.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_ratio", default=0, type=float, help="Linear warmup over warmup_ratio.")

    parser.add_argument("--logging_steps", type=int, default=5000, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=5000, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        # default = True,
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", default=True, help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=33, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

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
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))

    processor = processors[args.task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # proxies = {"http": "http://10.10.1.10:3128","https": "https://10.10.1.10:1080",}

    if args.task_name == 'arcch':
        args.logging_steps = 10
        args.save_steps = 10
        pretrained = "./saved_models/race_large"
        from CGR.data_arcch import AMRDataLoader
    elif args.task_name == 'obqa':
        args.logging_steps = 10
        args.save_steps = 10
        pretrained = './saved_models/arc_regents_large'
        from CGR.data_obqa import AMRDataLoader

    retriever_config = config_class_retriever.from_pretrained(
        args.retriever_name,
        cache_dir="./cache_file/",
        finetuning_task='retriever',
    )
    retriever_tokenizer = tokenizer_class_retriever.from_pretrained(
        args.retriever_name,
        cache_dir="./cache_file/",
        do_lower_case=True,
    )
    retriever = model_class_retriever.from_pretrained(
        args.retriever_name,
        cache_dir="./cache_file/",
        from_tf=bool(".ckpt" in args.retriever_name),
        config=retriever_config,
    )

    args.data_dir = './Data/begin/' + args.task_name
    args.output_dir = './saved_models/' + args.task_name + args.post
    reader_tokenizer = tokenizer_class_reader.from_pretrained(
        pretrained,
        cache_dir="./cache_file/",
        do_lower_case=args.do_lower_case, )
    reader_config = config_class_reader.from_pretrained(
        pretrained,
        cache_dir="./cache_file/",
        num_labels=num_labels, finetuning_task=args.task_name,
        pad_token_id=reader_tokenizer.pad_token_id, )

    reader = model_class_reader.from_pretrained(
        pretrained,
        cache_dir="./cache_file/",
        config=reader_config)
    # core fact
    core_f_addr = args.data_dir + "/core.dict"
    core_vec_addr = args.data_dir + "/core.npy"

    core_f = read_fact(core_f_addr) if os.path.exists(core_f_addr) else None
    core_vec = np.load(core_vec_addr)if os.path.exists(core_vec_addr) else None
    core_dict = read_amr(core_f_addr) if os.path.exists(core_f_addr) else None

    # common fact
    comm_f_addr = args.data_dir + '/' + args.task_name + ".dict"
    comm_vec_addr = args.data_dir + '/' + args.task_name + ".npy"
    comm_f = read_fact(comm_f_addr)
    comm_vec = np.load(comm_vec_addr)
    comm_dict = read_amr(comm_f_addr)
    fact_data = [core_f, core_vec, core_dict, comm_f, comm_vec, comm_dict]

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    retriever.to(args.device)
    reader.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    best_steps = 0

    # Training
    if args.do_train:
        # train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        # logger.info("evaluate before trainings")
        # result = evaluate(args, reader, retriever, reader_tokenizer, retriever_tokenizer, fact_data)#evaluate before training
        # evaluate(args, reader, retriever, reader_tokenizer, retriever_tokenizer, fact_data, test=True)
        global_step, tr_loss, best_steps = train(args, reader, retriever, reader_tokenizer, retriever_tokenizer, fact_data, AMRDataLoader)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            reader.module if hasattr(reader, "module") else reader
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        reader_tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned

        reader = model_class_reader.from_pretrained(args.output_dir)
        reader_tokenizer = tokenizer_class_reader.from_pretrained(args.output_dir)
        reader.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        if not args.do_train:
            args.output_dir = args.model_name
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            reader = model_class_reader.from_pretrained(checkpoint)
            reader.to(args.device)
            result = evaluate(args, reader, retriever, reader_tokenizer, retriever_tokenizer, fact_data, prefix=prefix, AMRDataLoader=AMRDataLoader)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    if args.do_test and args.local_rank in [-1, 0]:
        if not args.do_train:
            args.output_dir = args.model_name
        # checkpoints = [os.path.join(args.output_dir, x) for x in os.listdir(args.output_dir) if
        #                os.path.isdir(os.path.join(args.output_dir, x))]
        checkpoints = [args.output_dir]
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            reader = model_class_reader.from_pretrained(checkpoint)
            reader.to(args.device)
            result = evaluate(args, reader, retriever, reader_tokenizer, retriever_tokenizer, fact_data, prefix=prefix, test=True, AMRDataLoader=AMRDataLoader)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)
    if best_steps:
        logger.info("best steps of eval acc is the following checkpoints: %s", best_steps)
    return results


if __name__ == "__main__":
    main()
