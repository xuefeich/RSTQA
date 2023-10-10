import io, requests, zipfile
import os
import json
import argparse
from datetime import datetime
from tag_op import options
import torch
import torch.nn as nn
from pprint import pprint
from tag_op.tagop.util import create_logger, set_environment
from tag_op.data.tatqa_batch_gen import TaTQATestBatchGen
from tag_op.data.data_util import OPERATOR_CLASSES_,ARITHMETIC_CLASSES_
from tag_op.data.data_util import get_op_1, get_op_2, get_arithmetic_op_index_1, get_arithmetic_op_index_2
from tag_op.data.data_util import get_op_3, get_arithmetic_op_index_3
from transformers import RobertaModel, BertModel,TapasModel
from tag_op.tagop.modeling_rstqa import TagopModel
from tag_op.tagop.model import TagopPredictModel

parser = argparse.ArgumentParser("Tagop training task.")
options.add_data_args(parser)
options.add_bert_args(parser)
parser.add_argument("--eval_batch_size", type=int, default=32)
parser.add_argument("--model_path", type=str, default="checkpoint")
parser.add_argument("--tapas_path", type=str, default="")
parser.add_argument("--mode", type=int, default=1)
parser.add_argument("--op_mode", type=int, default=0)
parser.add_argument("--ablation_mode", type=int, default=0)
parser.add_argument("--encoder", type=str, default='roberta')
parser.add_argument("--test_data_dir", type=str, default="tag_op/cache/")
parser.add_argument("--num_ops", type=int, default=6)
parser.add_argument("--set", type=str, default="dev")
parser.add_argument("--plm_path", type=str, default='')
args = parser.parse_args()
if args.ablation_mode != 0:
    args.model_path = args.model_path + "_{}_{}".format(args.op_mode, args.ablation_mode)
if args.ablation_mode != 0:
    args.data_dir = args.data_dir + "_{}_{}".format(args.op_mode, args.ablation_mode)

if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

args.cuda = args.gpu_num > 0

logger = create_logger("TagOp Predictor", log_file=os.path.join(args.save_dir, args.log_file))

pprint(args)
set_environment(args.cuda)

def main():
    dev_itr = TaTQATestBatchGen(args, data_mode=args.set, encoder=args.encoder,num_ops = args.num_ops)
    # test_itr = TaTQATestBatchGen(args, data_mode="test", encoder=args.encoder)

    if args.encoder == 'roberta':
        bert_model = RobertaModel.from_pretrained(args.roberta_model)
    elif args.encoder == 'tapas':
        bert_model = TapasModel.from_pretrained(args.plm_path + "/tapas.large")
    elif args.encoder == 'deberta':
        bert_model = TapasModel.from_pretrained(args.plm_path + "/deberta-v2-xlarge")

    if args.ablation_mode == 0:
        operators = OPERATOR_CLASSES_
        arithmetic_op_index = [3, 4, 6, 7, 8, 9]
    elif args.ablation_mode == 1:
        operators = get_op_1(args.op_mode)
    elif args.ablation_mode == 2:
        operators = get_op_2(args.op_mode)
    else:
        operators = get_op_3(args.op_mode)

    if args.ablation_mode == 1:
        arithmetic_op_index = get_arithmetic_op_index_1(args.op_mode)
    elif args.ablation_mode == 2:
        arithmetic_op_index = get_arithmetic_op_index_2(args.op_mode)
    else:
        arithmetic_op_index = get_arithmetic_op_index_3(args.op_mode)


    # with open("ari_operator_ids.json",'r',encoding='utf-8') as fr:
    #     ari_operator_ids = json.load(fr)
    #     fr.close()
    # print(bert_model.config)
    # bert_model.resize_token_embeddings(bert_model.config.vocab_size+len(ari_operator_ids)+1)
    bert_model.resize_token_embeddings(bert_model.config.vocab_size+1)
    network = TagopModel(
        encoder = bert_model,
        config = bert_model.config,
        bsz = None,
        operator_classes = len(operators),
        ari_classes = len(ARITHMETIC_CLASSES_),
        scale_classes = 5,
        num_ops = args.num_ops,
        arithmetic_op_index = arithmetic_op_index,
        op_mode = args.op_mode,
        ablation_mode = args.ablation_mode,
        ari_operator_ids = ari_operator_ids,
    )
    network.load_state_dict(torch.load(os.path.join(args.model_path,"checkpoint_best.pt")))
    model = TagopPredictModel(args, network)
    logger.info("Below are the result on Dev set...")
    model.reset()
    model.avg_reset()
    pred_json = model.predict(dev_itr)
    json.dump(pred_json,open(os.path.join(args.save_dir, f"pred_result_on_{args.set}.json"), 'w'))
    model.get_metrics()
    print("pred file saved")
    '''
    print("text: "+ str(network.text_matched_numbers) + " / "+str(network.text_numbers))
    print("table: "+ str(network.table_matched_numbers) + " / "+str(network.table_numbers))
    print("table-text: "+ str(network.mix_matched_numbers) + " / "+str(network.mix_numbers))
    '''
    # logger.info("===========")
    # logger.info("Below are the result on Test set...")
    # model.reset()
    # model.avg_reset()
    # pred_json = model.predict(test_itr)
    # json.dump(pred_json, open(os.path.join(args.save_dir, 'pred_result_on_test.json'),'w'))
    # model.get_metrics()


if __name__ == "__main__":
    main()
