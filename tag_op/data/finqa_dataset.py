import re
import string
import json
from tqdm import tqdm
import random
import numpy as np

np.set_printoptions(threshold=np.inf)
import pandas as pd
from typing import List, Dict, Tuple
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from .file_utils import is_scatter_available
from tatqa_utils import *
from .finqa_data_util import *
from .derivation_split import infix_evaluator
from .mapping_split import find_mapping

const_dict = {1:1,2:2,3:3,4:4,5:5,100:6,1000:7,1000000:8}


if is_scatter_available():
    from torch_scatter import scatter


def convert_start_end_tags(split_tags, paragraph_index):
    in_split_tags = split_tags.copy()
    split_tags = [0 for i in range(len(split_tags))]
    for i in range(len(in_split_tags)):
        if in_split_tags[i] == 1:
            current_index = paragraph_index[i]
            split_tags[i] = 1
            paragraph_index_ = paragraph_index[i:]
            for j in range(1, len(paragraph_index_)):
                if paragraph_index_[j] == current_index:
                    split_tags[i + j] = 1
                else:
                    break
            break
    for i in range(1, len(in_split_tags)):
        if in_split_tags[-i] == 1:
            current_index = paragraph_index[-i]
            split_tags[-i] = 1
            paragraph_index_ = paragraph_index[:-i]
            for j in range(1, len(paragraph_index_)):
                if paragraph_index_[-j] == current_index:
                    split_tags[-i - j] = 1
                else:
                    break
            break
    del in_split_tags
    return split_tags


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or (len(c) == 1 and ord(c) == 0x202F):
        return True
    return False


def sortFunc(elem):
    return elem[1]


def get_order_by_tf_idf(question, paragraphs):
    sorted_order = []
    corpus = [question]
    for order, text in paragraphs.items():
        corpus.append(text)
        sorted_order.append(order)
    tf_idf = TfidfVectorizer().fit_transform(corpus)
    cosine_similarities = linear_kernel(tf_idf[0:1], tf_idf).flatten()[1:]
    sorted_similarities = sorted(enumerate(cosine_similarities), key=lambda x: x[1])
    idx = [i[0] for i in sorted_similarities][::-1]
    return [sorted_order[index] for index in idx]


def get_answer_nums(table_answer_coordinates: List, paragraph_answer_coordinates: Dict):
    if table_answer_coordinates is not None:
        table_answer_num = len(table_answer_coordinates)
    else:
        table_answer_num = 0
    paragraph_answer_nums = 0
    if paragraph_answer_coordinates:
        for value in paragraph_answer_coordinates.values():
            paragraph_answer_nums += len(value)
    return table_answer_num, paragraph_answer_nums


def get_operands_index(label_ids, token_type_ids):
    row_ids = token_type_ids[:, :, 2]
    column_ids = token_type_ids[:, :, 1]
    max_num_rows = 64
    max_num_columns = 32
    row_index = IndexMap(
        indices=torch.min(row_ids, torch.as_tensor(max_num_rows - 1, device=row_ids.device)),
        num_segments=max_num_rows,
        batch_dims=1,
    )
    col_index = IndexMap(
        indices=torch.min(column_ids, torch.as_tensor(max_num_columns - 1, device=column_ids.device)),
        num_segments=max_num_columns,
        batch_dims=1,
    )
    cell_index = ProductIndexMap(row_index, col_index).indices
    first_operand_start = torch.argmax((label_ids != 0).int(), dim=1)[0]
    label_ids = label_ids[0, first_operand_start:]
    cell_index_first = cell_index[0, first_operand_start:]
    first_operand_end = torch.argmax(((cell_index_first - cell_index[0, first_operand_start]) != 0).int())

    label_ids = label_ids[first_operand_end:]
    cell_index_first = cell_index_first[first_operand_end:]
    first_operand_end = first_operand_end + first_operand_start

    second_operand_start = torch.argmax((label_ids != 0).int())
    cell_index_second = cell_index_first[second_operand_start:]
    second_operand_end = torch.argmax(
        ((cell_index_second - cell_index_first[second_operand_start]) != 0).int()) + second_operand_start
    second_operand_start += first_operand_end
    second_operand_end += first_operand_end
    return first_operand_start, first_operand_end, second_operand_start, second_operand_end


def get_tokens_from_ids(ids, tokenizer):
    tokens = []
    sub_tokens = []
    for id in ids:
        token = tokenizer._convert_id_to_token(id)
        if len(sub_tokens) == 0:
            sub_tokens.append(token)
        elif str(token).startswith("##"):
            sub_tokens.append(token[2:])
        elif len(sub_tokens) != 0:
            tokens.append("".join(sub_tokens))
            sub_tokens = [token]
    tokens.append("".join(sub_tokens))
    return "".join(tokens)


def get_number_mask(table):
    max_num_rows = 64
    max_num_columns = 32
    columns = table.columns.tolist()
    number_mask = np.zeros((1, max_num_columns * max_num_rows))
    number_value = np.ones((1, max_num_columns * max_num_rows)) * np.nan
    for index, row in table.iterrows():
        for col_index in columns:
            col_index = int(col_index)
            in_cell_index = (index + 1) * max_num_columns + col_index + 1
            table_content = row[col_index]
            number = to_number(table_content)
            if number is not None:
                number_mask[0, in_cell_index] = 1
                number_value[0, in_cell_index] = float(number)
    return number_mask, number_value


def tokenize_answer(answer):
    answer_tokens = []
    prev_is_whitespace = True
    for i, c in enumerate(answer):
        if is_whitespace(c):  # or c in ["-", "–", "~"]:
            prev_is_whitespace = True
        elif c in ["-", "–", "~"]:
            answer_tokens.append(c)
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                answer_tokens.append(c)
            else:
                answer_tokens[-1] += c
            prev_is_whitespace = False
    return answer_tokens


def string_tokenizer(string: str, tokenizer) -> List[int]:
    if not string:
        return []
    tokens = []
    prev_is_whitespace = True
    for i, c in enumerate(string):
        if is_whitespace(c):  # or c in ["-", "–", "~"]:
            prev_is_whitespace = True
        elif c in ["-", "–", "~"]:
            tokens.append(c)
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                tokens.append(c)
            else:
                tokens[-1] += c
            prev_is_whitespace = False  # char_to_word_offset.append(len(tokens) - 1)

    split_tokens = []
    for i, token in enumerate(tokens):
        if i != 0:
            sub_tokens = tokenizer._tokenize(" " + token)
        else:
            sub_tokens = tokenizer._tokenize(token)

        for sub_token in sub_tokens:
            split_tokens.append(sub_token)

    ids = tokenizer.convert_tokens_to_ids(split_tokens)
    return ids


def table_tokenize(table, tokenizer):
    table_cell_tokens = []
    table_ids = []
    table_cell_index = []
    table_cell_number_value = []
    col_index = []
    cur_id = 1
    current_cell_index = 1
    for i in range(len(table)):
        for j in range(len(table[i])):
            cell_ids = string_tokenizer(table[i][j], tokenizer)
            if not cell_ids:
                continue
            table_ids += cell_ids
            if is_number(table[i][j]):
                table_cell_number_value.append(to_number(table[i][j]))
            else:
                table_cell_number_value.append(np.nan)
            col_index.append(cur_id)
            table_cell_tokens.append(table[i][j])
            table_cell_index += [current_cell_index for _ in range(len(cell_ids))]
            current_cell_index += 1
        cur_id += 1
    return table_cell_tokens, table_ids, table_cell_number_value, table_cell_index,col_index


def table_tagging(table, tokenizer, mapping):
    table_tags = []
    table_mapping = False
    answer_coordinates = None
    if "table" in mapping and len(mapping["table"]) != 0:
        table_mapping = True
        answer_coordinates = mapping["table"]
    for i in range(len(table)):
        for j in range(len(table[i])):
            cell_ids = string_tokenizer(table[i][j], tokenizer)
            if not cell_ids:
                continue
            if table_mapping:
                if [i, j] in answer_coordinates:
                    table_tags += [1 for _ in range(len(cell_ids))]
                else:
                    table_tags += [0 for _ in range(len(cell_ids))]
            else:
                table_tags += [0 for _ in range(len(cell_ids))]
    return table_tags
    

def paragraph_tokenize(question, paragraphs, tokenizer):
    paragraphs_copy = paragraphs.copy()
    paragraphs = {}
    for paragraph in paragraphs_copy:
        paragraphs[paragraph["order"]] = paragraph["text"]
    del paragraphs_copy
    split_tokens = []
    number_mask = []
    number_value = []
    tokens = []
    word_piece_mask = []
    paragraph_index = []
    # apply tf-idf to calculate text-similarity
    sorted_order = get_order_by_tf_idf(question, paragraphs)
    for order in sorted_order:
        text = paragraphs[order]
        prev_is_whitespace = True
        start_index = 0
        wait_add = False
        for i, c in enumerate(text):
            if is_whitespace(c):  # or c in ["-", "–", "~"]:
                if wait_add:
                    wait_add = False
                prev_is_whitespace = True
            elif c in ["-", "–", "~"]:
                if wait_add:
                    wait_add = False
                tokens.append(c)
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    tokens.append(c)
                    wait_add = True
                    start_index = i
                else:
                    tokens[-1] += c
                prev_is_whitespace = False
    current_token_index = 1
    for i, token in enumerate(tokens):
        if i != 0:
            sub_tokens = tokenizer._tokenize(" " + token)
        else:
            sub_tokens = tokenizer._tokenize(token)
        number = to_number(token)
        if number is not None:
            number_value.append(float(number))
        else:
            number_value.append(np.nan)
        for sub_token in sub_tokens:
            split_tokens.append(sub_token)
            paragraph_index.append(current_token_index)
        current_token_index += 1
        word_piece_mask += [1]
        if len(sub_tokens) > 1:
            word_piece_mask += [0] * (len(sub_tokens) - 1)
    paragraph_ids = tokenizer.convert_tokens_to_ids(split_tokens)
    return tokens, paragraph_ids, word_piece_mask, number_mask, number_value, paragraph_index,sorted_order



def paragraph_tagging(question, paragraphs, tokenizer, mapping,sorted_order):
    paragraphs_copy = paragraphs.copy()
    paragraphs = {}
    for paragraph in paragraphs_copy:
        paragraphs[paragraph["order"]] = paragraph["text"]
    del paragraphs_copy
    split_tokens = []
    split_tags = []
    number_mask = []
    number_value = []
    tokens = []
    tags = []
    word_piece_mask = []
    paragraph_index = []
    paragraph_mapping = False
    paragraph_mapping_orders = []
    if "paragraph" in list(mapping.keys()) and len(mapping["paragraph"].keys()) != 0:
        paragraph_mapping = True
        paragraph_mapping_orders = list(mapping["paragraph"].keys())
    for order in sorted_order:
        text = paragraphs[order]
        prev_is_whitespace = True
        answer_indexs = None
        if paragraph_mapping and str(order) in paragraph_mapping_orders:
            answer_indexs = mapping["paragraph"][str(order)]
        current_tags = [0 for i in range(len(text))]
        if answer_indexs is not None:
            for answer_index in answer_indexs:
                current_tags[answer_index[0]:answer_index[1]] = \
                    [1 for i in range(len(current_tags[answer_index[0]:answer_index[1]]))]

        start_index = 0
        wait_add = False
        for i, c in enumerate(text):
            if is_whitespace(c):  # or c in ["-", "–", "~"]:
                if wait_add:
                    if 1 in current_tags[start_index:i]:
                        tags.append(1)
                    else:
                        tags.append(0)
                    wait_add = False
                prev_is_whitespace = True
            elif c in ["-", "–", "~"]:
                if wait_add:
                    if 1 in current_tags[start_index:i]:
                        tags.append(1)
                    else:
                        tags.append(0)
                    wait_add = False
                tokens.append(c)
                tags.append(0)
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    tokens.append(c)
                    wait_add = True
                    start_index = i
                else:
                    tokens[-1] += c
                prev_is_whitespace = False
        if wait_add:
            if 1 in current_tags[start_index:len(text)]:
                tags.append(1)
            else:
                tags.append(0)
    try:
        assert len(tokens) == len(tags)
    except AssertionError:
        print(len(tokens), len(tags))
        input()
    current_token_index = 1
    for i, token in enumerate(tokens):
        if i != 0:
            sub_tokens = tokenizer._tokenize(" " + token)
        else:
            sub_tokens = tokenizer._tokenize(token)

        number = to_number(token)
        if number is not None:
            number_value.append(float(number))
        else:
            number_value.append(np.nan)
        for sub_token in sub_tokens:
            split_tags.append(tags[i])
            split_tokens.append(sub_token)
            paragraph_index.append(current_token_index)
        current_token_index += 1
        word_piece_mask += [1]
        if len(sub_tokens) > 1:
            word_piece_mask += [0] * (len(sub_tokens) - 1)
    return split_tags


def question_tokenizer(question_text, tokenizer):
    return string_tokenizer(question_text, tokenizer)


def _concat(question_ids,
            table_ids,
            table_cell_index,
            paragraph_ids,
            paragraph_index,
            cls,
            sep,
            opt,
            question_length_limitation,
            passage_length_limitation,
            max_pieces,
            num_ops,
            ari_tags,
            const):
    in_table_cell_index = table_cell_index.copy()
    in_paragraph_index = paragraph_index.copy()
    input_ids = torch.zeros([1, max_pieces])
    input_segments = torch.zeros_like(input_ids)
    paragraph_mask = torch.zeros_like(input_ids)
    paragraph_index = torch.zeros_like(input_ids)
    table_mask = torch.zeros_like(input_ids)
    question_mask = torch.zeros_like(input_ids)
    table_index = torch.zeros_like(input_ids)
    opt_mask = torch.zeros_like(input_ids)
    opt_index = torch.zeros_like(input_ids)
    opd_two_tags = torch.zeros([1, num_ops, input_ids.shape[1]])
    ari_round_tags = torch.zeros([1, num_ops, input_ids.shape[1]])
    ari_round_labels = torch.zeros([1, num_ops, input_ids.shape[1]])
    if question_length_limitation is not None:
        if len(question_ids) > question_length_limitation:
            question_ids = question_ids[:question_length_limitation]
    question_ids = [cls] + const + [sep] + question_ids + [sep]
    question_length = len(question_ids)
    question_mask[0, 10:question_length - 1] = 1
    table_length = len(table_ids)
    paragraph_length = len(paragraph_ids)
    if passage_length_limitation is not None:
        if len(table_ids) > passage_length_limitation:
            passage_ids = table_ids[:passage_length_limitation]
            table_length = passage_length_limitation
            paragraph_length = 0
        elif len(table_ids) + len(paragraph_ids) > passage_length_limitation:
            passage_ids = table_ids + [sep] + paragraph_ids
            passage_ids = passage_ids[:passage_length_limitation]
            table_length = len(table_ids)
            paragraph_length = passage_length_limitation - table_length
        else:
            passage_ids = table_ids + [sep] + paragraph_ids
            table_length = len(table_ids)
            paragraph_length = len(paragraph_ids) + 1
    else:
        passage_ids = table_ids + [sep] + paragraph_ids

    passage_ids = passage_ids + [sep] + num_ops * [opt] + [sep]

    input_ids[0, :question_length] = torch.from_numpy(np.array(question_ids))
    input_ids[0, question_length:question_length + len(passage_ids)] = torch.from_numpy(np.array(passage_ids))
    attention_mask = input_ids != 0
    table_mask[0, question_length:question_length + table_length] = 1
    table_index[0, question_length:question_length + table_length] = \
        torch.from_numpy(np.array(in_table_cell_index[:table_length]))

    if paragraph_length > 1:
        paragraph_mask[0, question_length + table_length + 1:question_length + table_length + paragraph_length] = 1
        paragraph_index[0, question_length + table_length + 1:question_length + table_length + paragraph_length] = \
            torch.from_numpy(np.array(in_paragraph_index[:paragraph_length - 1]))

    opt_mask[0,question_length + table_length + paragraph_length + 1: question_length + table_length + paragraph_length + num_ops + 1] = 1
    if ari_tags != None:
        ari_table_tags = ari_tags["table"]
        ari_para_tags = ari_tags["para"]
        for i in range(num_ops):
            r_num_ops = len(ari_table_tags)
            if i >= r_num_ops:
                break
            if isinstance(ari_table_tags[i], dict):
                opd1_tags = np.array(ari_table_tags[i]["operand1"][:table_length])
                opd2_tags = np.array(ari_table_tags[i]["operand2"][:table_length])
                ari_round_tags[0,i,question_length:question_length + table_length] = torch.from_numpy(opd1_tags)
                opd_two_tags[0, i, question_length:question_length + table_length] = torch.from_numpy(opd2_tags)
                for j in np.where(opd2_tags == 1):
                    opd1_tags[j] = 1
                ari_round_labels[0, i, question_length:question_length + table_length] = torch.from_numpy(opd1_tags)

                if paragraph_length > 1:
                    p_opd1_tags = np.array(ari_para_tags[i]["operand1"][:paragraph_length - 1])
                    p_opd2_tags = np.array(ari_para_tags[i]["operand2"][:paragraph_length - 1])
                    ari_round_tags[0,i, question_length + table_length + 1:question_length + table_length + paragraph_length] = torch.from_numpy(p_opd1_tags)
                    opd_two_tags[0, i,question_length + table_length + 1:question_length + table_length + paragraph_length] = torch.from_numpy(p_opd2_tags)
                    for j in np.where(p_opd2_tags == 1):
                        p_opd1_tags[j] = 1
                    ari_round_labels[0, i,question_length + table_length + 1:question_length + table_length + paragraph_length] = torch.from_numpy(p_opd1_tags)

            else:
                ari_round_tags[0, i, question_length:question_length + table_length] = torch.from_numpy(
                    np.array(ari_table_tags[i][:table_length]))
                ari_round_labels[0, i, question_length:question_length + table_length] = torch.from_numpy(
                    np.array(ari_table_tags[i][:table_length]))
                if paragraph_length > 1:
                    ari_round_tags[0, i,
                    question_length + table_length + 1:question_length + table_length + paragraph_length] = torch.from_numpy(
                        np.array(ari_para_tags[i][:paragraph_length - 1]))
                    ari_round_labels[0, i,
                    question_length + table_length + 1:question_length + table_length + paragraph_length] = torch.from_numpy(
                        np.array(ari_para_tags[i][:paragraph_length - 1]))

    del in_table_cell_index
    del in_paragraph_index

    return input_ids, attention_mask, paragraph_mask, paragraph_index, table_mask, table_index, \
        input_segments, opt_mask, opt_index, ari_round_tags, opd_two_tags, ari_round_labels, question_mask


def _test_concat(question_ids,
            table_ids,
            table_cell_index,
            paragraph_ids,
            paragraph_index,
            cls,
            sep,
            opt,
            question_length_limitation,
            passage_length_limitation,
            max_pieces,
            num_ops,
            const):
    in_table_cell_index = table_cell_index.copy()
    in_paragraph_index = paragraph_index.copy()
    input_ids = torch.zeros([1, max_pieces])
    input_segments = torch.zeros_like(input_ids)
    paragraph_mask = torch.zeros_like(input_ids)
    paragraph_index = torch.zeros_like(input_ids)
    table_mask = torch.zeros_like(input_ids)
    question_mask = torch.zeros_like(input_ids)
    table_index = torch.zeros_like(input_ids)
    opt_mask = torch.zeros_like(input_ids)
    opt_index = torch.zeros_like(input_ids)
    if question_length_limitation is not None:
        if len(question_ids) > question_length_limitation:
            question_ids = question_ids[:question_length_limitation]
    question_ids = [cls] + const + [sep] + question_ids + [sep]
    question_length = len(question_ids)
    question_mask[0, 10:question_length - 1] = 1
    table_length = len(table_ids)
    paragraph_length = len(paragraph_ids)
    if passage_length_limitation is not None:
        if len(table_ids) > passage_length_limitation:
            passage_ids = table_ids[:passage_length_limitation]
            table_length = passage_length_limitation
            paragraph_length = 0
        elif len(table_ids) + len(paragraph_ids) > passage_length_limitation:
            passage_ids = table_ids + [sep] + paragraph_ids
            passage_ids = passage_ids[:passage_length_limitation]
            table_length = len(table_ids)
            paragraph_length = passage_length_limitation - table_length
        else:
            passage_ids = table_ids + [sep] + paragraph_ids
            table_length = len(table_ids)
            paragraph_length = len(paragraph_ids) + 1
    else:
        passage_ids = table_ids + [sep] + paragraph_ids

    passage_ids = passage_ids + [sep] + num_ops * [opt] + [sep]

    input_ids[0, :question_length] = torch.from_numpy(np.array(question_ids))
    input_ids[0, question_length:question_length + len(passage_ids)] = torch.from_numpy(np.array(passage_ids))
    attention_mask = input_ids != 0
    table_mask[0, question_length:question_length + table_length] = 1
    table_index[0, question_length:question_length + table_length] = \
        torch.from_numpy(np.array(in_table_cell_index[:table_length]))

    if paragraph_length > 1:
        paragraph_mask[0, question_length + table_length + 1:question_length + table_length + paragraph_length] = 1
        paragraph_index[0, question_length + table_length + 1:question_length + table_length + paragraph_length] = \
            torch.from_numpy(np.array(in_paragraph_index[:paragraph_length - 1]))

    opt_mask[0,question_length + table_length + paragraph_length + 1: question_length + table_length + paragraph_length + num_ops + 1] = 1
    del in_table_cell_index
    del in_paragraph_index
    return input_ids, attention_mask, paragraph_mask, paragraph_index, table_mask, table_index, \
        input_segments, opt_mask, opt_index,question_mask

def combine_tags(tags1, tags2):
    cat_tags = torch.cat((tags1[0], tags2[0]), dim=0)
    sum_tags = torch.sum(cat_tags, dim=0)
    com_tags = torch.where(sum_tags > 0, 1, 0)
    whole_tags = torch.zeros([1, 512])
    whole_tags[0] = com_tags
    return whole_tags


class FinqaTrainReader(object):
    def __init__(self, tokenizer,
                 passage_length_limit: int = None, question_length_limit: int = None, sep="<s>", op_mode: int = 8,
                 ablation_mode: int = 0, num_ari_ops: int = 6):
        self.max_pieces = 512
        self.tokenizer = tokenizer
        self.passage_length_limit = passage_length_limit
        self.question_length_limit = question_length_limit
        self.sep = self.tokenizer._convert_token_to_id(sep)
        self.cls = self.tokenizer._convert_token_to_id("<s>")
        self.opt = self.tokenizer._convert_token_to_id("<OPT>")
        self.const = []
        for const in list(const_dict.keys()):
            self.const.append(self.tokenizer._convert_token_to_id(str(const)))
                     
        self.num_ops = num_ari_ops
        tokens = self.tokenizer._tokenize("Feb 2 Nov")
        self.skip_count = 0
        self.op_skip = 0
        self.ari_skip = 0
        self.op_mode = op_mode

    def _make_instance(self, input_ids, attention_mask, token_type_ids, paragraph_mask, table_mask,
                       paragraph_number_value, table_cell_number_value, paragraph_index, table_cell_index,
                       paragraph_tokens, table_cell_tokens, answer_dict, question_id, opt_mask, question_mask,col_index):

        opt_id = torch.nonzero(opt_mask == 1)[0, 1]

        return {
            "input_ids": np.array(input_ids),
            "attention_mask": np.array(attention_mask),
            "token_type_ids": np.array(token_type_ids),
            "paragraph_mask": np.array(paragraph_mask),
            "table_mask": np.array(table_mask),
            "question_mask": np.array(question_mask),
            "paragraph_number_value": np.array(paragraph_number_value),
            "table_cell_number_value": np.array(table_cell_number_value),
            "col_index" : np.array(col_index),
            "paragraph_index": np.array(paragraph_index),
            "table_cell_index": np.array(table_cell_index),
            "paragraph_tokens": paragraph_tokens,
            "table_cell_tokens": table_cell_tokens,
            "answer_dict": answer_dict,
            "question_id": question_id,
            "opt_mask": opt_id,
        }

    def _to_instance(self, question: str, table: List[List[str]], paragraphs: List[Dict], answer: str, derivation: str,question_id: str):
        question_text = question.strip()
        table_cell_tokens, table_ids,table_cell_number_value, table_cell_index,col_index = table_tokenize(table, self.tokenizer)
        paragraph_tokens, paragraph_ids,  paragraph_word_piece_mask, paragraph_number_mask, \
            paragraph_number_value, paragraph_index,sorted_order = paragraph_tokenize(question, paragraphs, self.tokenizer)

        order_labels = np.zeros(self.num_ops)
        ari_ops = [0] * self.num_ops
        opt_labels = torch.zeros(1, self.num_ops - 1, self.num_ops - 1)
        ari_tags = {'table': [], 'para': []}
        const_keys = list(const_dict.keys())
        const_list = []
        const_labels = []
        number_indexes = []
        if answer in ["yes","no"]:
            task = 0
            d = derivation.split("),")[0]
            [opd1,opd2] = d.split(",")
            [op,opd1] = opd1.split("(")
            op = op.strip(" ")
            opd2 = opd2.strip(")")
            n1 = False
            n2 = False
            if "const" in opd1:
                try:
                   opd1n = int(opd1.strip(" ").strip("const_"))
                except:
                   return None
                if opd1n not in const_keys:
                   return None
                if opd1n not in const_list:
                    number_indexes.append([const_dict[opd1n]]+[0]*9)
                    const_labels.append([0,0,0,0,0,0])
                    const_labels[-1][i] = 1
                    const_list.append(opd1n)
                else:
                    const_labels[const_list.index(opd1n)][i] = 1
            else:
                n1 = True
                opd1_mapping = find_mapping(opd1,table,paragraphs)
                if opd1_mapping is None:
                  return None
            if "const" in opd2:
                try:
                   opd2n = int(opd2.strip(" ").strip("const_"))
                except:
                   return None
                if opd2n not in const_keys:
                   return None
                if opd2n not in const_list:
                    number_indexes.append([const_dict[opd2n]]+[0]*9)
                    const_labels.append([0,0,0,0,0,0])
                    const_labels[-1][i] = 1
                    const_list.append(opd2n)
                else:
                    const_labels[const_list.index(opd2n)][i] = 1
            else:
                n2 = True
                opd2_mapping = find_mapping(opd2,table,paragraphs)
                if opd2_mapping is None:
                  return None
            op1_table_tags = table_tagging(table, self.tokenizer, opd1_mapping)
            op1_para_tags = paragraph_tagging(question, paragraphs, self.tokenizer, opd1_mapping,sorted_order)
            op2_table_tags = table_tagging(table, self.tokenizer, opd2_mapping)
            op2_para_tags = paragraph_tagging(question, paragraphs, self.tokenizer, opd2_mapping,sorted_order)
            ari_tags['table'].append({"operand1": op1_table_tags, "operand2": op2_table_tags})
            ari_tags['para'].append({"operand1": op1_para_tags, "operand2": op2_para_tags})
            if "const" in opd1:
                if "const" in opd2:
                    if int(opd1.strip(" ").strip("const_")) > int(opd2.strip(" ").strip("const_")):
                        order_labels[0] = 1
            elif n1 and "table" in opd1_mapping:
                if "const" in opd2:
                    order_labels[0] = 1
                elif n2 and "table" in opd2_mapping:
                    if opd1_mapping["table"][0][0] >  opd2_mapping["table"][0][0]:
                        order_labels[0] = 1
                    elif opd1_mapping["table"][0][1] >  opd2_mapping["table"][0][1]:
                        order_labels[0] = 1
            elif n1 and "paragraph" in opd1_mapping:
                if n2 and "paragraph" in opd2_mapping:
                    opd1_pid = sorted_order.index(list(opd1_mapping["paragraph"].keys())[0])
                    opd2_pid = sorted_order.index(list(opd2_mapping["paragraph"].keys())[0])
                    if int(opd1_pid) > int(opd2_pid):
                        order_labels[0] = 1
                    elif opd1_pid == opd2_pid and opd1_mapping["paragraph"][opd1_pid][0][0] > opd2_mapping["paragraph"][opd2_pid][0][0]:
                        order_labels[0] = 1
                else:
                    order_labels[0] = 1
        else:
            task = 1
            dlist = derivation.split("),")
            for i,d in enumerate(dlist):
                if i >= self.num_ops:
                    break
                [opd1,opd2] = d.split(",")
                [op,opd1] = opd1.split("(")
                op = op.strip(" ")
                ari_ops[i] = ARI_CLASSES_[op]
                
                if "table_" in op:
                   for i in range(len(table)):
                      cell = table[i][0]
                      if opd1.strip(" ") in cell:
                         temp_table_tags = table_tagging(table, self.tokenizer, {"table":[[i,0]]})
                         ari_tags['table'].append(temp_table_tags)
                         ari_tags['para'].append([0]*512)
                         break
                   continue

                opd2 = opd2.strip(")")
                n1 = False
                n2 = False
                if "#" in opd1:
                    j = int(opd1.strip(" ").strip("#"))
                    if j > 5:
                      return None
                    opt_labels[0,j,i-1] = 1
                elif "const" in opd1:
                    try:
                       opd1n = int(opd1.strip(" ").strip("const_"))
                    except:
                       return None
                    if opd1n not in const_keys:
                        return None
                    if opd1n not in const_list:
                        number_indexes.append([const_dict[opd1n]]+[0]*9)
                        const_labels.append([0,0,0,0,0,0])
                        const_labels[-1][i] = 1
                        const_list.append(opd1n)
                    else:
                        const_labels[const_list.index(opd1n)][i] = 1
                else:
                    n1 = True
                    opd1_mapping = find_mapping(opd1,table,paragraphs)                                
                    if opd1_mapping is None:
                       return None
                if "#" in opd2:
                    j = int(opd2.strip(" ").strip("#"))
                    if j > 5:
                      return None
                    if op in ["add","multiply"]:
                        opt_labels[0,j,i-1] = 1
                    else:
                        opt_labels[0, j, i - 1] = 2
                elif "const" in opd2:
                    try:
                       opd2n = int(opd2.strip(" ").strip("const_"))
                    except:
                       return None
                    if opd2n not in const_keys:
                        return None
                    if opd2n not in const_list:
                        number_indexes.append([const_dict[opd2n]]+[0]*9)
                        const_labels.append([0,0,0,0,0,0])
                        const_labels[-1][i] = 1
                        const_list.append(opd2n)
                    else:
                        const_labels[const_list.index(opd2n)][i] = 1
                else:
                    n2 = True
                    opd2_mapping = find_mapping(opd2,table,paragraphs)
                    if opd2_mapping is None:
                       return None
                if op in ["add", "multiply"]:
                    order_labels[i] = -100
                    if n1 and "table" in opd1_mapping:
                        if n2 and "table" in opd2_mapping:
                            opd1_mapping["table"] += opd2_mapping["table"]
                    elif n1:
                        if n2 and "table" in opd2_mapping:
                            opd1_mapping["table"] = opd2_mapping["table"]
                    elif n2 and "table" in opd2_mapping:
                       opd1_mapping = opd2_mapping
                    if n1 and "paragraph" in opd1_mapping:
                        if n2 and "paragraph" in opd2_mapping:
                            pid = list(opd2_mapping["paragraph"].keys())[0]
                            if pid in opd1_mapping["paragraph"]:
                                opd1_mapping["paragraph"][pid] += opd2_mapping["paragraph"][pid]
                            else:
                                opd1_mapping["paragraph"][pid] = opd2_mapping["paragraph"][pid]
                    elif n1:
                        if n2 and "paragraph" in opd2_mapping:
                            opd1_mapping["paragraph"] = opd2_mapping["paragraph"]
                    elif n2 and "paragraph" in opd2_mapping:
                         opd1_mapping = opd2_mapping

                    if n1 or n2:
                        temp_para_tags = paragraph_tagging(question, paragraphs, self.tokenizer, opd1_mapping,sorted_order)
                        temp_table_tags = table_tagging(table, self.tokenizer, opd1_mapping)
                        ari_tags['table'].append(temp_table_tags)
                        ari_tags['para'].append(temp_para_tags)
                    else:
                        return None
                else:
                    op1_table_tags = [0]*512
                    op2_table_tags = [0]*512
                    op1_para_tags = [0]*512
                    op2_para_tags = [0]*512
                    if n1:
                       op1_table_tags = table_tagging(table, self.tokenizer, opd1_mapping)
                       op1_para_tags = paragraph_tagging(question, paragraphs, self.tokenizer, opd1_mapping,sorted_order)
                    if n2:
                       op2_table_tags = table_tagging(table, self.tokenizer, opd2_mapping)
                       op2_para_tags = paragraph_tagging(question, paragraphs, self.tokenizer, opd2_mapping,sorted_order)
                    ari_tags['table'].append({"operand1": op1_table_tags, "operand2": op2_table_tags})
                    ari_tags['para'].append({"operand1": op1_para_tags, "operand2": op2_para_tags})
                    if "const" in opd1:
                       if "const" in opd2:
                           if int(opd1.strip(" ").strip("const_")) > int(opd2.strip(" ").strip("const_")):
                               order_labels[i] = 1
                    elif n1 and "table" in opd1_mapping:
                        if "const" in opd2:
                            order_labels[i] = 1
                        elif n2 and "table" in opd2_mapping:
                            if opd1_mapping["table"][0][0] >  opd2_mapping["table"][0][0]:
                                order_labels[i] = 1
                            elif opd1_mapping["table"][0][1] >  opd2_mapping["table"][0][1]:
                                order_labels[i] = 1
                    elif n1 and "paragraph" in opd1_mapping:
                        if n2 and "paragraph" in opd2_mapping:
                            pid1 = list(opd1_mapping["paragraph"].keys())[0]
                            pid2 = list(opd2_mapping["paragraph"].keys())[0]
                            opd1_pid = sorted_order.index(pid1)
                            opd2_pid = sorted_order.index(pid2)
                            if int(opd1_pid) > int(opd2_pid):
                                order_labels[i] = 1
                            elif opd1_pid == opd2_pid and opd1_mapping["paragraph"][pid1][0][0] > opd2_mapping["paragraph"][pid2][0][0]:
                                order_labels[i] = 1
                        else:
                            order_labels[i] = 1
        
        for i in range(len(table)):
            for j in range(len(table[i])):
                if table[i][j] == '' or table[i][j] == 'N/A' or table[i][j] == 'n/a':
                    table[i][j] = "NONE"
        table = pd.DataFrame(table, dtype=np.str_)
        column_relation = {}
        for column_name in table.columns.values.tolist():
            column_relation[column_name] = str(column_name)
        table.rename(columns=column_relation, inplace=True)

        question_ids = question_tokenizer(question_text, self.tokenizer)

        input_ids, attention_mask, paragraph_mask, paragraph_index, \
            table_mask, table_index, token_type_ids, opt_mask, opt_index, ari_round_tags, opd_two_tags, ari_round_labels, question_mask = \
            _concat(question_ids, table_ids, table_cell_index,
                    paragraph_ids, paragraph_index, self.cls,
                    self.sep, self.opt, self.question_length_limit,
                    self.passage_length_limit, self.max_pieces, self.num_ops, ari_tags,self.const)
        
        tags = combine_tags(ari_round_tags,opd_two_tags)

        ari_round_labels = torch.where(tags > 0, ari_round_labels, -100)
        answer_dict = {"answer": answer}
        return self._make_instance(input_ids, attention_mask, token_type_ids, paragraph_mask, table_mask,
                                   paragraph_number_value, table_cell_number_value, paragraph_index, table_index, tags,
                                   task,paragraph_tokens, table_cell_tokens, answer_dict, question_id, ari_ops, opt_mask,
                                   opt_labels, ari_round_labels, order_labels, question_mask,const_labels,number_indexes,const_list,col_index)

    def _read(self, file_path: str):
        print("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)
            dataset_file.close()
        instances = []
        for one in tqdm(dataset):
            table = one['table_ori']
            paragraphs = one['pre_text'] + one["post_text"]
            para_dicts = []
            for od, p in enumerate(paragraphs):
                para_dicts.append({"order": od + 1, "text": p})
            question = one['qa']["question"].strip()
            answer = one['qa']["answer"]
            exe_ans = one['qa']["exe_ans"]
            derivation = one['qa']["program"]
            if isinstance(exe_ans,float):
                exe_ans = str(np.round(exe_ans,4))
            try:
               instance = self._to_instance(question, table, para_dicts, str(exe_ans), derivation, one["id"])
            except:
               continue
            if instance is not None:
                instances.append(instance)
        return instances


class FinqaTestReader(object):
    def __init__(self, tokenizer,
                 passage_length_limit: int = None, question_length_limit: int = None, sep="<s>", op_mode: int = 8,
                 ablation_mode: int = 0, num_ari_ops: int = 6):
        self.max_pieces = 512
        self.tokenizer = tokenizer
        self.passage_length_limit = passage_length_limit
        self.question_length_limit = question_length_limit
        self.sep = self.tokenizer._convert_token_to_id(sep)
        self.cls = self.tokenizer._convert_token_to_id("<s>")
        self.opt = self.tokenizer._convert_token_to_id("<OPT>")
        self.const = []
        for const in list(const_dict.keys()):
            self.const.append(self.tokenizer._convert_token_to_id(str(const)))
                     
        self.num_ops = num_ari_ops
        tokens = self.tokenizer._tokenize("Feb 2 Nov")
        self.skip_count = 0
        self.op_skip = 0
        self.ari_skip = 0
        self.op_mode = op_mode

    def _make_instance(self, input_ids, attention_mask, token_type_ids, paragraph_mask, table_mask,
                       paragraph_number_value, table_cell_number_value, paragraph_index, table_cell_index,
                       tags_ground_truth, operator_ground_truth, paragraph_tokens,
                       table_cell_tokens, answer_dict, question_id, ari_ops, opt_mask, opt_labels,
                       ari_labels, order_labels, question_mask,const_labels,const_indexes,const_list,col_index):

        if ari_ops != None:
            if ari_ops == [0] * self.num_ops:
                print("no ari ops")
                ari_ops = [-100] * self.num_ops
            else:
                get0 = False
                for i in range(self.num_ops):
                    if get0 == True:
                        ari_ops[i] = -100
                        order_labels[i] = -100
                    if ari_ops[i] == 0:
                        order_labels[i] = -100
                        if get0 == False:
                            get0 = True
        else:
            ari_ops = [-100] * self.num_ops

        if ari_ops == [-100] * self.num_ops:
            opt_labels[0, :, :] = -100
            ari_labels[0, :, :] = -100
            order_labels[1:] = -100
            number_indexes = []
            ari_sel_labels = []
            if ari_ops[1] == 0:
                opt_labels[0, :, :] = -100
        else:
            for i in range(1, self.num_ops - 1):
                for j in range(i):
                    opt_labels[0, i, j] = -100
                if ari_ops[i] == 0:
                    ari_labels[0, i:, :] = -100
                    opt_labels[0, i - 1:, :] = -100
                    opt_labels[0, :, i - 1:] = -100

            number_indexes = []
            cur_indexes = []
            cur = 0
            selected_indexes = torch.nonzero(tags_ground_truth[0]).squeeze(-1)
            if len(selected_indexes) == 0:
                print("no number")
                return None
            for sel in selected_indexes:
                sel = int(sel)
                if int(table_cell_index[0, sel]) != 0:
                    if int(table_cell_index[0, sel]) == cur or cur_indexes == []:
                        if cur_indexes == []:
                            cur = int(table_cell_index[0, sel])
                        cur_indexes.append(sel)
                    else:
                        cur = int(table_cell_index[0, sel])
                        number_indexes.append(cur_indexes)
                        cur_indexes = [sel]
                else:
                    if int(paragraph_index[0, sel]) == cur or cur_indexes == []:
                        if cur_indexes == []:
                            cur = int(paragraph_index[0, sel])
                        cur_indexes.append(sel)
                    else:
                        cur = int(paragraph_index[0, sel])
                        number_indexes.append(cur_indexes)
                        cur_indexes = [sel]
            number_indexes.append(cur_indexes)
            distinct_si = []
            for i, ni in enumerate(number_indexes):
                distinct_si.append(ni[0])
                p = 10 - len(ni)
                if p > 0:
                    number_indexes[i] += [0] * p
                else:
                    number_indexes[i] = number_indexes[i][:10]
                    print("long number")
                    print(ni)
                    print(table_cell_index[0, ni])
                    print(paragraph_index[0, ni])
                    if int(table_cell_index[0, ni[0]]) != 0:
                        print(table_cell_number_value[int(table_cell_index[0, ni[0]]) - 1])
                    elif int(paragraph_index[0, ni[0]]) != 0:
                        print(paragraph_number_value[int(paragraph_index[0, ni[0]]) - 1])
                    else:
                        print("extract err")  # if question_answer["uid"] in ignore_ids:

            ari_sel_labels = ari_labels[0, :, distinct_si].transpose(0, 1)
            if len(const_labels) > 0:
               number_indexes  = const_indexes + number_indexes
               const_labels = torch.from_numpy(np.array(const_labels))
               ari_sel_labels = torch.cat((const_labels,ari_sel_labels),dim = 0)
               for const in const_list:
                   tags_ground_truth[0,const_dict[int(const)]] = 1
            if ari_sel_labels.shape[0] != len(number_indexes):
                print(ari_sel_labels)
                print(number_indexes)
                exit(0)
        opt_id = torch.nonzero(opt_mask == 1)[0, 1]

        return {
            "input_ids": np.array(input_ids),
            "attention_mask": np.array(attention_mask),
            "token_type_ids": np.array(token_type_ids),
            "paragraph_mask": np.array(paragraph_mask),
            "table_mask": np.array(table_mask),
            "question_mask": np.array(question_mask),
            "paragraph_number_value": np.array(paragraph_number_value),
            "table_cell_number_value": np.array(table_cell_number_value),
            "col_index" : np.array(col_index),
            "paragraph_index": np.array(paragraph_index),
            "table_cell_index": np.array(table_cell_index),
            "tag_labels": np.array(tags_ground_truth),
            "operator_label": int(operator_ground_truth),
            "paragraph_tokens": paragraph_tokens,
            "table_cell_tokens": table_cell_tokens,
            "answer_dict": answer_dict,
            "question_id": question_id,
            "ari_ops": torch.LongTensor(ari_ops),
            "opt_mask": opt_id,
            "order_labels": torch.LongTensor(order_labels),
            "ari_labels": torch.LongTensor(np.array(ari_sel_labels)),
            "selected_indexes": np.array(number_indexes),
            "opt_labels": torch.LongTensor(np.array(opt_labels)),
        }

    def _to_instance(self, question: str, table: List[List[str]], paragraphs: List[Dict], answer: str, derivation: str,question_id: str):
        question_text = question.strip()
        table_cell_tokens, table_ids,table_cell_number_value, table_cell_index,col_index = table_tokenize(table, self.tokenizer)
        paragraph_tokens, paragraph_ids,  paragraph_word_piece_mask, paragraph_number_mask, \
            paragraph_number_value, paragraph_index,sorted_order = paragraph_tokenize(question, paragraphs, self.tokenizer)

        
        for i in range(len(table)):
            for j in range(len(table[i])):
                if table[i][j] == '' or table[i][j] == 'N/A' or table[i][j] == 'n/a':
                    table[i][j] = "NONE"
        table = pd.DataFrame(table, dtype=np.str_)
        column_relation = {}
        for column_name in table.columns.values.tolist():
            column_relation[column_name] = str(column_name)
        table.rename(columns=column_relation, inplace=True)

        question_ids = question_tokenizer(question_text, self.tokenizer)

        input_ids, attention_mask, paragraph_mask, paragraph_index, \
            table_mask, table_index, token_type_ids, opt_mask, opt_index, question_mask = \
            _test_concat(question_ids, table_ids, table_cell_index,
                    paragraph_ids, paragraph_index, self.cls,
                    self.sep, self.opt, self.question_length_limit,
                    self.passage_length_limit, self.max_pieces, self.num_ops, self.const)
        answer_dict = {"answer": answer}
        return self._make_instance(input_ids, attention_mask, token_type_ids, paragraph_mask, table_mask,
                                   paragraph_number_value, table_cell_number_value, paragraph_index, table_index,
                                   paragraph_tokens, table_cell_tokens, answer_dict, question_id,opt_mask, question_mask,col_index)

    def _read(self, file_path: str):
        print("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)
            dataset_file.close()
        instances = []
        for one in tqdm(dataset):
            table = one['table_ori']
            paragraphs = one['pre_text'] + one["post_text"]
            para_dicts = []
            for od, p in enumerate(paragraphs):
                para_dicts.append({"order": od + 1, "text": p})
            question = one['qa']["question"].strip()
            answer = one['qa']["answer"]
            exe_ans = one['qa']["exe_ans"]
            derivation = one['qa']["program"]
            if isinstance(exe_ans,float):
                exe_ans = str(np.round(exe_ans,4))
            try:
               instance = self._to_instance(question, table, para_dicts, str(exe_ans), derivation, one["id"])
            except:
               continue
            if instance is not None:
                instances.append(instance)
        return instances



# ### Beginning of everything related to segmented tensors ###

class IndexMap(object):
    """Index grouping entries within a tensor."""

    def __init__(self, indices, num_segments, batch_dims=0):
        """
        Creates an index
        Args:
            indices (:obj:`torch.LongTensor`, same shape as a `values` Tensor to which the indices refer):
                Tensor containing the indices.
            num_segments (:obj:`torch.LongTensor`):
                Scalar tensor, the number of segments. All elements in a batched segmented tensor must have the same
                number of segments (although many segments can be empty).
            batch_dims (:obj:`int`, `optional`, defaults to 0):
                The number of batch dimensions. The first `batch_dims` dimensions of a SegmentedTensor are treated as
                batch dimensions. Segments in different batch elements are always distinct even if they have the same
                index.
        """
        self.indices = torch.as_tensor(indices)
        self.num_segments = torch.as_tensor(num_segments, device=indices.device)
        self.batch_dims = batch_dims

    def batch_shape(self):
        return self.indices.size()[: self.batch_dims]  # returns a torch.Size object


class ProductIndexMap(IndexMap):
    """The product of two indices."""

    def __init__(self, outer_index, inner_index):
        """
        Combines indices i and j into pairs (i, j). The result is an index where each segment (i, j) is the
        intersection of segments i and j. For example if the inputs represent table cells indexed by respectively rows
        and columns the output will be a table indexed by (row, column) pairs, i.e. by cell. The implementation
        combines indices {0, .., n - 1} and {0, .., m - 1} into {0, .., nm - 1}. The output has `num_segments` equal to
        `outer_index.num_segments` * `inner_index.num_segments`
        Args:
            outer_index (:obj:`IndexMap`):
                IndexMap.
            inner_index (:obj:`IndexMap`):
                IndexMap, must have the same shape as `outer_index`.
        """
        if outer_index.batch_dims != inner_index.batch_dims:
            raise ValueError("outer_index.batch_dims and inner_index.batch_dims must be the same.")

        super(ProductIndexMap, self).__init__(
            indices=(inner_index.indices + outer_index.indices * inner_index.num_segments),
            num_segments=inner_index.num_segments * outer_index.num_segments,
            batch_dims=inner_index.batch_dims,
        )
        self.outer_index = outer_index
        self.inner_index = inner_index

    def project_outer(self, index):
        """Projects an index with the same index set onto the outer components."""
        return IndexMap(
            indices=(index.indices // self.inner_index.num_segments).type(torch.float).floor().type(torch.long),
            num_segments=self.outer_index.num_segments,
            batch_dims=index.batch_dims,
        )

    def project_inner(self, index):
        """Projects an index with the same index set onto the inner components."""
        return IndexMap(
            indices=torch.fmod(index.indices, self.inner_index.num_segments)
            .type(torch.float)
            .floor()
            .type(torch.long),
            num_segments=self.inner_index.num_segments,
            batch_dims=index.batch_dims,
        )


def reduce_mean(values, index, name="segmented_reduce_mean"):
    """
    Averages a tensor over its segments.
    Outputs 0 for empty segments.
    This operations computes the mean over segments, with support for:
        - Batching using the first dimensions [B1, B2, ..., Bn]. Each element in a batch can have different indices.
        - Vectorization using the last dimension [V1, V2, ...]. If they are present, the output will be a mean of
          vectors rather than scalars.
    Only the middle dimensions [I1, ..., Ik] are reduced by the operation.
    Args:
        values (:obj:`torch.Tensor` of shape [B1, B2, ..., Bn, I1, .., Ik, V1, V2, ..]):
            Tensor containing the values of which the mean must be taken segment-wise.
        index (:obj:`IndexMap`, indices are of shape [B1, B2, ..., Bn, I1, .., Ik].):
            Index defining the segments.
        name (:obj:`str`, `optional`, defaults to 'segmented_reduce_sum'):
            Name for the operation. Currently not used
    Returns:
        output_values (:obj:`torch.Tensor`of shape [B1, B2, ..., Bn, num_segments, V1, V2, ..]): Tensor containing the
        output values. output_index (:obj:`IndexMap`): IndexMap with shape [B1, B2, ..., Bn, num_segments].
    """
    return _segment_reduce(values, index, "mean", name)


def flatten(index, name="segmented_flatten"):
    """
    Flattens a batched index map (which is typically of shape batch_size, seq_length) to a 1d index map. This operation
    relabels the segments to keep batch elements distinct. The k-th batch element will have indices shifted by
    `num_segments` * (k - 1). The result is a tensor with `num_segments` multiplied by the number of elements in the
    batch.
    Args:
        index (:obj:`IndexMap`):
            IndexMap to flatten.
        name (:obj:`str`, `optional`, defaults to 'segmented_flatten'):
            Name for the operation. Currently not used
    Returns:
        (:obj:`IndexMap`): The flattened IndexMap.
    """
    # first, get batch_size as scalar tensor
    batch_size = torch.prod(torch.tensor(list(index.batch_shape())))
    # next, create offset as 1-D tensor of length batch_size,
    # and multiply element-wise by num segments (to offset different elements in the batch) e.g. if batch size is 2: [0, 64]
    offset = torch.arange(start=0, end=batch_size, device=index.num_segments.device) * index.num_segments
    offset = offset.view(index.batch_shape())
    for _ in range(index.batch_dims, len(index.indices.size())):  # typically range(1,2)
        offset = offset.unsqueeze(-1)

    indices = offset + index.indices
    return IndexMap(indices=indices.view(-1), num_segments=index.num_segments * batch_size, batch_dims=0)


def range_index_map(batch_shape, num_segments, name="range_index_map"):
    """
    Constructs an index map equal to range(num_segments).
    Args:
        batch_shape (:obj:`torch.Size`):
            Batch shape
        num_segments (:obj:`int`):
            Number of segments
        name (:obj:`str`, `optional`, defaults to 'range_index_map'):
            Name for the operation. Currently not used
    Returns:
        (:obj:`IndexMap`): IndexMap of shape batch_shape with elements equal to range(num_segments).
    """
    batch_shape = torch.as_tensor(
        batch_shape, dtype=torch.long
    )  # create a rank 1 tensor vector containing batch_shape (e.g. [2])
    assert len(batch_shape.size()) == 1
    num_segments = torch.as_tensor(num_segments)  # create a rank 0 tensor (scalar) containing num_segments (e.g. 64)
    assert len(num_segments.size()) == 0

    indices = torch.arange(
        start=0, end=num_segments, device=num_segments.device
    )  # create a rank 1 vector with num_segments elements
    new_tensor = torch.cat(
        [torch.ones_like(batch_shape, dtype=torch.long, device=num_segments.device), num_segments.unsqueeze(dim=0)],
        dim=0,
    )
    # new_tensor is just a vector of [1 64] for example (assuming only 1 batch dimension)
    new_shape = [int(x) for x in new_tensor.tolist()]
    indices = indices.view(new_shape)

    multiples = torch.cat([batch_shape, torch.as_tensor([1])], dim=0)
    indices = indices.repeat(multiples.tolist())
    # equivalent (in Numpy:)
    # indices = torch.as_tensor(np.tile(indices.numpy(), multiples.tolist()))

    return IndexMap(indices=indices, num_segments=num_segments, batch_dims=list(batch_shape.size())[0])


def _segment_reduce(values, index, segment_reduce_fn, name):
    """
    Applies a segment reduction segment-wise.
    Args:
        values (:obj:`torch.Tensor`):
            Tensor with segment values.
        index (:obj:`IndexMap`):
            IndexMap.
        segment_reduce_fn (:obj:`str`):
            Name for the reduce operation. One of "sum", "mean", "max" or "min".
        name (:obj:`str`):
            Name for the operation. Currently not used
    Returns:
        (:obj:`IndexMap`): IndexMap of shape batch_shape with elements equal to range(num_segments).
    """
    # Flatten the batch dimensions, as segments ops (scatter) do not support batching.
    # However if `values` has extra dimensions to the right keep them
    # unflattened. Segmented ops support vector-valued operations.
    flat_index = flatten(index)
    vector_shape = values.size()[len(index.indices.size()):]  # torch.Size object
    flattened_shape = torch.cat(
        [torch.as_tensor([-1], dtype=torch.long), torch.as_tensor(vector_shape, dtype=torch.long)], dim=0
    )
    # changed "view" by "reshape" in the following line
    flat_values = values.reshape(flattened_shape.tolist())

    segment_means = scatter(
        src=flat_values,
        index=flat_index.indices.type(torch.long),
        dim=0,
        dim_size=flat_index.num_segments,
        reduce=segment_reduce_fn,
    )

    # Unflatten the values.
    new_shape = torch.cat(
        [
            torch.as_tensor(index.batch_shape(), dtype=torch.long),
            torch.as_tensor([index.num_segments], dtype=torch.long),
            torch.as_tensor(vector_shape, dtype=torch.long),
        ],
        dim=0,
    )

    output_values = segment_means.view(new_shape.tolist())
    output_index = range_index_map(index.batch_shape(), index.num_segments)
    return output_values, output_index
