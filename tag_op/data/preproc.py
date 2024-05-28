import re
import string
import json
from tqdm import tqdm
import random
import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd
from typing import List, Dict, Tuple

from tatqa_utils import  *
from .data_util import *
from .derivation_split import infix_evaluator
class TagTaTQAReader(object):
    def __init__(self):
        print("start")
    def _make_instance(self, question,table,paragraphs,outputs,ari_ops,operands,scale,id):

        return {
            "question" : question,
            "table" : table,
            "text" : paragraphs,
            "outputs" : outputs,
            "operator" : ari_ops,
            "operand" : operands,
            "scale":scale,
            "question_id" : id
        }

    def _to_instance(self, question: str, table: List[List[str]], paragraphs: List[Dict], answer_from: str,
                     answer_type: str, answer: str, derivation: str, facts: list, answer_mapping: Dict, scale: str,
                     question_id: str):

        f = 1

        if answer_type == "arithmetic":
            dvt_split_suc = 0
            try:
                ari_operations = infix_evaluator(derivation)
                dvt_split_suc = 1
            except:
                print("derivation split err")
                f = 0
            if dvt_split_suc == 1:
                ari_ops = ';'.join([i[0] for i in ari_operations])
                operands = ';'.join([','.join(i[1:]) for i in ari_operations])
                outputs = derivation + ";" + answer

        elif answer_type == "count":
            ari_ops = ""
            operands = ""
            outputs = ','.join(facts)+";"+answer
        else:
            ari_ops = ""
            operands = ""
            outputs = answer

        if f == 0:
            return None

        return self._make_instance(question,table,paragraphs,outputs,ari_ops,operands,scale,question_id)

    def _read(self, file_path: str):
        print("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)
            dataset_file.close()
        # f = open("3round.json",'w')
        instances = []
        key_error_count = 0
        index_error_count = 0
        assert_error_count = 0
        count = 0
        for one in tqdm(dataset):
            table = one['table']['table']
            paragraphs = one['paragraphs']
            questions = one['questions']

            for question_answer in questions:
                try:
                    question = question_answer["question"].strip()
                    answer_type = question_answer["answer_type"]
                    derivation = question_answer["derivation"]
                    answer = question_answer["answer"]
                    answer_mapping = question_answer["mapping"]
                    facts = question_answer["facts"]
                    answer_from = question_answer["answer_from"]
                    scale = question_answer["scale"]
                except RuntimeError as e:
                    print(f"run time error:{e}")
                except KeyError:
                    key_error_count += 1
                    print("KeyError. Total Error Count: {}".format(key_error_count))
                except IndexError:
                    index_error_count += 1
                    print("IndexError. Total Error Count: {}".format(index_error_count))
                except AssertionError:
                    assert_error_count += 1
                    print("AssertError. Total Error Count: {}".format(assert_error_count))
                instance = self._to_instance(question, table, paragraphs, answer_from,
                                             answer_type, answer, derivation, facts, answer_mapping, scale,
                                             question_answer["uid"])
                if instance is not None:
                    instances.append(instance)
        return instances
