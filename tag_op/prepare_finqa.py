import os
import pickle
import argparse
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from transformers.models.bert import BertTokenizer

import json
parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, default='./dataset_finqa')
parser.add_argument("--model_path", type=str, default='./model')
parser.add_argument("--output_dir", type=str, default="./tag_op/finqa")
parser.add_argument("--passage_length_limit", type=int, default=447)
parser.add_argument("--question_length_limit", type=int, default=46)
parser.add_argument("--encoder", type=str, default="roberta")
parser.add_argument("--mode", type=str, default='train')
parser.add_argument("--num_arithmetic_operators",type=int,default=6)

args = parser.parse_args()

if args.encoder == 'roberta':
    from tag_op.data.finqa_dataset import FinqaTrainReader, FinqaTestReader
    tokenizer = RobertaTokenizer.from_pretrained(args.model_path + "/roberta.large")
    sep = '<s>'
    #tokenizer.add_special_tokens({'additional_special_tokens':['<OPT>','<STP>','<SUM>','<DIFF>','<DIVIDE>','<TIMES>','<AVERAGE>']})
    tokenizer.add_special_tokens({'additional_special_tokens':['<OPT>']})
elif args.encoder == 'deberta':
    from transformers import DebertaV2Tokenizer
    from tag_op.data.finqa_dataset import TagTaTQAReader, TagTaTQATestReader
    tokenizer = DebertaV2Tokenizer.from_pretrained(args.model_path + "/deberta-v2-xlarge")
    tokenizer.add_special_tokens({'additional_special_tokens':['[OPT]']})
    sep = '[SEP]'

if args.mode == 'test':
    data_reader = FinqaTestReader(tokenizer, args.passage_length_limit, args.question_length_limit, sep=sep,num_ari_ops = args.num_arithmetic_operators)
    data_mode = ["test"]
elif args.mode == 'dev':
    data_reader = FinqaTestReader(tokenizer, args.passage_length_limit, args.question_length_limit, sep=sep,num_ari_ops = args.num_arithmetic_operators)
    data_mode = ["dev"]
else:
    data_reader = FinqaTrainReader(tokenizer, args.passage_length_limit, args.question_length_limit, sep=sep,num_ari_ops = args.num_arithmetic_operators)
    data_mode = ["train"]

data_format = "{}.json"
print(f'==== NOTE ====: encoder:{args.encoder}, mode:{args.mode}')

for dm in data_mode:
    dpath = os.path.join(args.input_path, data_format.format(dm))
    data = data_reader._read(dpath)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if dm == "dev":
        data = data_reader._read(dpath)
    else:
        data = data_reader._read(dpath)
    print("Save data to {}.".format(os.path.join(args.output_dir, f"finqa_{args.encoder}_cached_{dm}.pkl")))
    with open(os.path.join(args.output_dir, f"finqa_{args.encoder}_cached_{dm}.pkl"), "wb") as f:
        pickle.dump(data, f)
