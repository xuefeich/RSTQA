import io, requests, zipfile
import os
import json
import argparse
from datetime import datetime
from tag_op import options
import torch
import torch.nn as nn
from pprint import pprint
from tag_op.data.finqa_data_util import ARI_CLASSES_
from tag_op.tagop.util import create_logger, set_environment
from tag_op.data.finqa_batch_gen import TaTQABatchGen, TaTQATestBatchGen
from transformers import RobertaModel, BertModel
from tag_op.tagop.modeling_finqa import TagopModel
from tag_op.tagop.model import TagopFineTuningModel
from pathlib import Path
parser = argparse.ArgumentParser("TagOp training task.")
options.add_data_args(parser)
options.add_train_args(parser)
options.add_bert_args(parser)
parser.add_argument("--encoder", type=str, default='roberta')
parser.add_argument("--op_mode", type=int, default=0)
parser.add_argument("--ablation_mode", type=int, default=0)
parser.add_argument("--test_data_dir", type=str, default="./tag_op/cache")
parser.add_argument("--num_ops", type=int, default=6)

args = parser.parse_args()
if args.ablation_mode != 0:
    args.save_dir = args.save_dir + "_{}_{}".format(args.op_mode, args.ablation_mode)
    args.data_dir = args.data_dir + "_{}_{}".format(args.op_mode, args.ablation_mode)

Path(args.save_dir).mkdir(parents=True, exist_ok=True)

args.cuda = args.gpu_num > 0
args_path = os.path.join(args.save_dir, "args.json")
with open(args_path, "w") as f:
    json.dump((vars(args)), f)

args.batch_size = args.batch_size // args.gradient_accumulation_steps

logger = create_logger("Roberta Training", log_file=os.path.join(args.save_dir, args.log_file))

pprint(args)
set_environment(args.seed, args.cuda)

def main():
    print(torch.cuda.device_count())
    best_result = float("-inf")
    logger.info("Loading data...")
    train_itr = TaTQABatchGen(args, data_mode = "train", encoder=args.encoder,num_ops = args.num_ops)
    dev_itr = TaTQATestBatchGen(args, data_mode="dev", encoder=args.encoder,num_ops = args.num_ops)

    num_train_steps = int(args.max_epoch * len(train_itr) / args.gradient_accumulation_steps)
    logger.info("Num update steps {}!".format(num_train_steps))

    logger.info(f"Build {args.encoder} model.")
    if args.encoder == 'bert':
        bert_model = BertModel.from_pretrained('bert-large-uncased')
    elif args.encoder == 'roberta':
        bert_model = RobertaModel.from_pretrained(args.roberta_model)
    elif args.encoder == 'finbert':
        bert_model = BertModel.from_pretrained(args.finbert_model)

    if args.ablation_mode == 0:
        operators = OPERATOR_CLASSES_
    elif args.ablation_mode == 1:
        operators = get_op_1(args.op_mode)
    elif args.ablation_mode == 2:
        operators = get_op_2(args.op_mode)
    else:
        operators = get_op_3(args.op_mode)
    if args.ablation_mode == 0:
        arithmetic_op_index = [3, 4, 6, 7, 8, 9]
    elif args.ablation_mode == 1:
        arithmetic_op_index = get_arithmetic_op_index_1(args.op_mode)
    elif args.ablation_mode == 2:
        arithmetic_op_index = get_arithmetic_op_index_2(args.op_mode)
    else:
        arithmetic_op_index = get_arithmetic_op_index_3(args.op_mode)

    print(bert_model.config)
    bert_model.resize_token_embeddings(bert_model.config.vocab_size+1)
    network = TagopModel(
        encoder = bert_model,
        config = bert_model.config,
        bsz = args.batch_size,
        ari_classes = len(ARI_CLASSES_),
        num_ops = args.num_ops,
        operator_criterion = nn.CrossEntropyLoss(reduction = "sum"),
        operand_criterion = nn.CrossEntropyLoss(),
        task_criterion = nn.CrossEntropyLoss(),
        opt_criterion = nn.CrossEntropyLoss(reduction = "sum"),
        order_criterion = nn.CrossEntropyLoss(reduction = "sum"),
    )
    logger.info("Build optimizer etc...")

    parameters = filter(lambda p:p.reuires_grad,network.parameters())
    model = TagopFineTuningModel(args, network, num_train_steps=num_train_steps)
    train_start = datetime.now()
    first = True
    for epoch in range(1, args.max_epoch + 1):
        model.reset()
        if not first:
            train_itr.reset()
        first = False
        logger.info('At epoch {}'.format(epoch))
        #model.predict(dev_itr)
        #exit(0)
        for step, batch in enumerate(train_itr):
            model.update(batch,epoch = epoch)
            if model.step % (args.log_per_updates * args.gradient_accumulation_steps) == 0 or model.step == 1:
                logger.info("Updates[{0:6}] train loss[{1:.5f}] remaining[{2}].\r\n".format(
                    model.updates, model.train_loss.avg,
                    str((datetime.now() - train_start) / (step + 1) * (num_train_steps - step - 1)).split('.')[0]))
                model.avg_reset()
        model.reset()
        model.avg_reset()
        model.predict(dev_itr)
        metrics = model.get_metrics(logger)
        model.avg_reset()
        if metrics["f1"] > best_result:
            save_prefix = os.path.join(args.save_dir, "checkpoint_best")
            model.save(save_prefix, epoch)
            best_result = metrics["f1"]
            logger.info("Best eval F1 {} at epoch {}.\r\n".format(best_result, epoch))
if __name__ == "__main__":
    main()
