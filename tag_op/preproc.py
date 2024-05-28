import os
import argparse
import json
from tag_op.data.preproc import TagTaTQAReader
parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, default='./dataset_tagop')
parser.add_argument("--output_dir", type=str, default="./tag_op/cache")
parser.add_argument("--mode", type=str, default='train')

args = parser.parse_args()

if args.mode == 'test':
    data_reader = TagTaTQAReader()
    data_mode = ["test"]
elif args.mode == 'dev':
    data_reader = TagTaTQAReader()
    data_mode = ["dev"]
else:
    data_reader = TagTaTQAReader()
    data_mode = ["train"]

data_format = "tatqa_dataset_{}.json"
for dm in data_mode:
    dpath = os.path.join(args.input_path, data_format.format(dm))
    data = data_reader._read(dpath)
    print("Save data to {}.".format(os.path.join(args.output_dir, f"new_cached_{dm}.json")))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, f"new_cached_{dm}.json"), "w") as f:
        for d in data:
          f.write(json.dumps(d,ensure_ascii=False)+"\n")
        f.close()
