import os
import pickle
import argparse
from tag_op.data.hqa_dataset import TagTaTQAReader, TagTaTQATestReader
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer

from transformers.models.bert import BertTokenizer

import json
parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, default='./HQA')
parser.add_argument("--output_dir", type=str, default="./tag_op/cache")
parser.add_argument("--passage_length_limit", type=int, default=458)
parser.add_argument("--question_length_limit", type=int, default=46)
parser.add_argument("--encoder", type=str, default="roberta")
parser.add_argument("--mode", type=str, default='train')
parser.add_argument("--num_arithmetic_operators",type=int,default=6)

args = parser.parse_args()
args.passage_length_limit = 462 - args.num_arithmetic_operators
if args.encoder == 'roberta':
    tokenizer = RobertaTokenizer.from_pretrained(args.input_path + "/roberta.large")
    sep = '<s>'
    #tokenizer.add_special_tokens({'additional_special_tokens':['<OPT>','<STP>','<SUM>','<DIFF>','<DIVIDE>','<TIMES>','<AVERAGE>']})
    tokenizer.add_special_tokens({'additional_special_tokens':['<OPT>']})
elif args.encoder == 'bert':
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    sep = '[SEP]'


if args.mode == 'test':
    data_reader = TagTaTQATestReader(tokenizer, args.passage_length_limit, args.question_length_limit, sep=sep,num_ari_ops = args.num_arithmetic_operators,mode = args.mode)
    data_mode = ["test"]
elif args.mode == 'dev':
    data_reader = TagTaTQATestReader(tokenizer, args.passage_length_limit, args.question_length_limit, sep=sep,num_ari_ops = args.num_arithmetic_operators,mode = args.mode)
    data_mode = ["dev"]
else:
    data_reader = TagTaTQAReader(tokenizer, args.passage_length_limit, args.question_length_limit, sep=sep,num_ari_ops = args.num_arithmetic_operators)
    data_mode = ["train"]

data_format = "hqa_{}.json"
print(f'==== NOTE ====: encoder:{args.encoder}, mode:{args.mode}')
'''
with open("ari_operator_ids.json",'w',encoding = 'utf-8') as fr:
    json.dump({"<STP>":tokenizer.encode('<STP>')[1],"<SUM>":tokenizer.encode('<SUM>')[1],"<DIFF>":tokenizer.encode('<DIFF>')[1],"<DIVIDE>":tokenizer.encode('<DIVIDE>')[1],"<TIMES>":tokenizer.encode('<TIMES>')[1],"<AVERAGE>":tokenizer.encode('<AVERAGE>')[1]},fr)
    fr.close()
'''
with open("ari_operator_ids.json",'w',encoding = 'utf-8') as fr:
    json.dump({"<OPT>":tokenizer.encode('<OPT>')[1]},fr)
    fr.close()
for dm in data_mode:
    dpath = os.path.join(args.input_path, data_format.format(dm))

    if dm == "dev":
        data,maxround_data = data_reader._read(dpath)
    else:
        data = data_reader._read(dpath)
    print(data_reader.skip_count)
    print(data_reader.ari_skip)
    print(data_reader.op_skip)
    data_reader.skip_count = 0
    print("Save data to {}.".format(os.path.join(args.output_dir, f"hqa_{args.encoder}_cached_{dm}.pkl")))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, f"hqa_{args.encoder}_cached_{dm}.pkl"), "wb") as f:
        pickle.dump(data, f)
    if dm == "dev":
        with open(os.path.join(args.output_dir, f"hqa_{args.encoder}_cached_{dm}_3round.pkl"), "wb") as fmax:
            pickle.dump(maxround_data, fmax)
