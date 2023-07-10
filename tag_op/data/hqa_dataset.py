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
from tatqa_utils import  *
from .hqa_data_util import *
from .hqa_data_util import  _is_average, _is_change_ratio, _is_diff, _is_division, _is_sum, _is_times
from .hqa_derivation_split import infix_evaluator
from .hqa_mapping_split import split_mapping
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
                    split_tags[i+j] = 1
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
                    split_tags[-i-j] = 1
                else:
                    break
            break
    del in_split_tags
    return split_tags

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
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
    sorted_similarities = sorted(enumerate(cosine_similarities), key=lambda x:x[1])
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
    first_operand_start = torch.argmax((label_ids!=0).int(), dim=1)[0]
    label_ids = label_ids[0, first_operand_start:]
    cell_index_first = cell_index[0, first_operand_start:]
    first_operand_end = torch.argmax(((cell_index_first-cell_index[0, first_operand_start])!=0).int())

    label_ids = label_ids[first_operand_end:]
    cell_index_first = cell_index_first[first_operand_end:]
    first_operand_end = first_operand_end+first_operand_start

    second_operand_start = torch.argmax((label_ids!=0).int())
    cell_index_second = cell_index_first[second_operand_start:]
    second_operand_end = torch.argmax(((cell_index_second-cell_index_first[second_operand_start])!=0).int())+second_operand_start
    second_operand_start+=first_operand_end
    second_operand_end+=first_operand_end
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
    number_mask = np.zeros((1, max_num_columns*max_num_rows))
    number_value = np.ones((1, max_num_columns*max_num_rows)) * np.nan
    for index, row in table.iterrows():
        for col_index in columns:
            col_index = int(col_index)
            in_cell_index = (index+1)*max_num_columns+col_index+1
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

def table_tokenize(table, tokenizer, mapping, answer_type):
    table_cell_tokens = []
    table_ids = []
    table_tags = []
    table_cell_index = []
    table_cell_number_value = []
    table_mapping = False
    answer_coordinates = None

    if "table" in mapping and len(mapping["table"]) != 0:
        table_mapping = True
        answer_coordinates = mapping["table"]

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
            table_cell_tokens.append(table[i][j])
            if table_mapping:
                if [i, j] in answer_coordinates:
                    table_tags += [1 for _ in range(len(cell_ids))]
                else:
                    table_tags += [0 for _ in range(len(cell_ids))]
            else:
                table_tags += [0 for _ in range(len(cell_ids))]
            table_cell_index += [current_cell_index for _ in range(len(cell_ids))]
            current_cell_index += 1
    return table_cell_tokens, table_ids, table_tags, table_cell_number_value, table_cell_index

def table_test_tokenize(table, tokenizer, mapping, answer_type):
    mapping_content = []
    table_cell_tokens = []
    table_ids = []
    table_tags = []
    table_cell_index = []
    table_cell_number_value = []
    table_mapping = False
    answer_coordinates = None

    if mapping and "table" in mapping and len(mapping["table"]) != 0:
        table_mapping = True
        answer_coordinates = mapping["table"]

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
            table_cell_tokens.append(table[i][j])
            if table_mapping:
                if [i, j] in answer_coordinates:
                    mapping_content.append(table[i][j])
                    table_tags += [1 for _ in range(len(cell_ids))]
                else:
                    table_tags += [0 for _ in range(len(cell_ids))]
            else:
                table_tags += [0 for _ in range(len(cell_ids))]
            table_cell_index += [current_cell_index for _ in range(len(cell_ids))]
            current_cell_index += 1
    return table_cell_tokens, table_ids, table_tags, table_cell_number_value, table_cell_index


def paragraph_tokenize(question, paragraphs, tokenizer, mapping, answer_type):
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
    # apply tf-idf to calculate text-similarity
    sorted_order = get_order_by_tf_idf(question, paragraphs)
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
        current_token_index+=1
        word_piece_mask += [1]
        if len(sub_tokens) > 1:
            word_piece_mask += [0] * (len(sub_tokens) - 1)
    paragraph_ids = tokenizer.convert_tokens_to_ids(split_tokens)
    return tokens, paragraph_ids, split_tags, word_piece_mask, number_mask, number_value, paragraph_index

def paragraph_test_tokenize(question, paragraphs, tokenizer, mapping, answer_type):
    mapping_content = []
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
    if mapping and "paragraph" in list(mapping.keys()) and len(mapping["paragraph"].keys()) != 0:
        paragraph_mapping = True
        paragraph_mapping_orders = list(mapping["paragraph"].keys())
    # apply tf-idf to calculate text-similarity
    sorted_order = get_order_by_tf_idf(question, paragraphs)
    for order in sorted_order:
        text = paragraphs[order]
        prev_is_whitespace = True
        answer_indexs = None
        if paragraph_mapping and str(order) in paragraph_mapping_orders:
            answer_indexs = mapping["paragraph"][str(order)]
        current_tags = [0 for i in range(len(text))]
        if answer_indexs is not None:
            for answer_index in answer_indexs:
                mapping_content.append(text[answer_index[0]:answer_index[1]])
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
        current_token_index+=1
        word_piece_mask += [1]
        if len(sub_tokens) > 1:
            word_piece_mask += [0] * (len(sub_tokens) - 1)
    paragraph_ids = tokenizer.convert_tokens_to_ids(split_tokens)
    return tokens, paragraph_ids, split_tags, word_piece_mask, number_mask, number_value, \
           paragraph_index, mapping_content

def question_tokenize(question, tokenizer, mapping, answer_type):
    split_tokens = []
    split_tags = []
    number_mask = []
    number_value = []
    tokens = []
    tags = []
    word_piece_mask = []
    question_index = []
    mapping_content = []

    question_mapping = False
    if "question" in list(mapping.keys()) and len(mapping["question"]) != 0:
        question_mapping = True
    current_tags = [0 for i in range(len(question))]
    prev_is_whitespace = True

    if question_mapping == True:
        for answer_index in mapping["question"]:
            mapping_content.append(question[answer_index[0]:answer_index[1]])
            current_tags[answer_index[0]:answer_index[1]] = \
                [1 for i in range(len(current_tags[answer_index[0]:answer_index[1]]))]
        
    start_index = 0
    wait_add = False
    for i, c in enumerate(question):
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
        if 1 in current_tags[start_index:len(question)]:
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
            question_index.append(current_token_index)
        current_token_index+=1
        word_piece_mask += [1]
        if len(sub_tokens) > 1:
            word_piece_mask += [0] * (len(sub_tokens) - 1)
    question_ids = tokenizer.convert_tokens_to_ids(split_tokens)
    return tokens, question_ids, split_tags, word_piece_mask, number_mask, number_value, question_index




def question_tokenizer(question_text, tokenizer):
    return string_tokenizer(question_text, tokenizer)


def _concat(question_ids,
            question_tags, 
            question_index,
            table_ids,
            table_tags,
            table_cell_index,
            #table_cell_number_value,
            paragraph_ids,
            paragraph_tags,
            paragraph_index,
            #paragraph_number_value,
            sep,
            opt,
            question_length_limitation,
            passage_length_limitation,
            max_pieces,
            num_ops,
            ari_tags,):
    in_table_cell_index = table_cell_index.copy()
    in_paragraph_index = paragraph_index.copy()
    in_question_index = question_index.copy()
    input_ids = torch.zeros([1, max_pieces])
    input_segments = torch.zeros_like(input_ids)
    paragraph_mask = torch.zeros_like(input_ids)
    paragraph_index = torch.zeros_like(input_ids)
    table_mask = torch.zeros_like(input_ids)
    question_mask = torch.zeros_like(input_ids)
    table_index = torch.zeros_like(input_ids)
    question_index = torch.zeros_like(input_ids)
    opt_mask = torch.zeros_like(input_ids)
    opt_index = torch.zeros_like(input_ids)
    tags = torch.zeros_like(input_ids)
    #ari_round_tags = torch.zeros([1,num_ops,input_ids.shape[1]])
    ari_round_labels = torch.zeros([1,num_ops,input_ids.shape[1]])
    if question_length_limitation is not None:
        if len(question_ids) > question_length_limitation:
            question_ids = question_ids[:question_length_limitation]
    question_ids = [sep] + question_ids + [sep]
    question_length = len(question_ids)
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

    question_mask[0,1:question_length-1] = 1
    question_index[0, 1:question_length-1] = \
        torch.from_numpy(np.array(in_question_index[:question_length-2]))
    tags[0, 1:question_length-1] = torch.from_numpy(np.array(question_tags[:question_length-2]))

    
    table_mask[0, question_length:question_length + table_length] = 1
    table_index[0, question_length:question_length + table_length] = \
        torch.from_numpy(np.array(in_table_cell_index[:table_length]))
    tags[0, question_length:question_length + table_length] = torch.from_numpy(np.array(table_tags[:table_length]))


    if paragraph_length > 1:
        paragraph_mask[0, question_length + table_length + 1:question_length + table_length + paragraph_length] = 1
        paragraph_index[0, question_length + table_length + 1:question_length + table_length + paragraph_length] = \
            torch.from_numpy(np.array(in_paragraph_index[:paragraph_length - 1]))
        tags[0, question_length + table_length + 1:question_length + table_length + paragraph_length] = \
            torch.from_numpy(np.array(paragraph_tags[:paragraph_length - 1]))

    opt_mask[0,question_length + table_length + paragraph_length + 1 : question_length + table_length + paragraph_length + num_ops+1] = 1
    if ari_tags != None:
      ari_question_tags = ari_tags["question"]
      ari_table_tags = ari_tags["table"]
      ari_para_tags = ari_tags["para"]
      for i in range(num_ops):
         r_num_ops = max(len(ari_table_tags),len(ari_para_tags),len(ari_question_tags))
         if i >= r_num_ops:
             break
         if isinstance(ari_table_tags[i],dict):
             opd1_tags = np.array(ari_table_tags[i]["operand1"][:table_length])
             opd2_tags = np.array(ari_table_tags[i]["operand2"][:table_length])
             for j in np.where(opd2_tags == 1):
                 opd1_tags[j] = 1
             ari_round_labels[0,i,question_length:question_length + table_length] = torch.from_numpy(opd1_tags)


             q_opd1_tags = np.array(ari_question_tags[i]["operand1"][:question_length-2])
             q_opd2_tags = np.array(ari_question_tags[i]["operand2"][:question_length-2])
             for j in np.where(q_opd2_tags == 1):
                 q_opd1_tags[j] = 1
             ari_round_labels[0,i,1:question_length-1] = torch.from_numpy(q_opd1_tags)

             if paragraph_length > 1:
                p_opd1_tags = np.array(ari_para_tags[i]["operand1"][:paragraph_length-1])
                p_opd2_tags = np.array(ari_para_tags[i]["operand2"][:paragraph_length-1])
                for j in np.where(p_opd2_tags == 1):
                    p_opd1_tags[j] = 1
                ari_round_labels[0,i, question_length + table_length + 1:question_length + table_length + paragraph_length] = torch.from_numpy(p_opd1_tags)

         else:
             ari_round_labels[0,i, 1:question_length-1] = torch.from_numpy(np.array(ari_question_tags[i][:question_length - 2]))
             ari_round_labels[0,i, question_length:question_length + table_length] = torch.from_numpy(np.array(ari_table_tags[i][:table_length]))
             if paragraph_length > 1:
                ari_round_labels[0,i, question_length + table_length + 1:question_length + table_length + paragraph_length] = torch.from_numpy(np.array(ari_para_tags[i][:paragraph_length-1]))

    del in_table_cell_index
    del in_paragraph_index
    del in_question_index

    return input_ids, attention_mask, paragraph_mask, paragraph_index, table_mask,  table_index, tags, \
            input_segments,opt_mask,opt_index,ari_round_labels,question_mask,question_index
def _test_concat(question_ids,
                 question_tags,
                 question_index,
                table_ids,
                table_tags,
                table_cell_index,
                #table_cell_number_value,
                paragraph_ids,
                paragraph_tags,
                paragraph_index,
                #paragraph_number_value,
                sep,
                opt,
                question_length_limitation,
                passage_length_limitation,
                max_pieces,
                num_ops):
    in_table_cell_index = table_cell_index.copy()
    in_paragraph_index = paragraph_index.copy()
    in_question_index = question_index.copy()
    input_ids = torch.zeros([1, max_pieces])
    input_segments = torch.zeros_like(input_ids)
    paragraph_mask = torch.zeros_like(input_ids)
    paragraph_index = torch.zeros_like(input_ids)
    table_mask = torch.zeros_like(input_ids)
    table_index = torch.zeros_like(input_ids)
    question_index = torch.zeros_like(input_ids)
    tags = torch.zeros_like(input_ids)
    question_mask = torch.zeros_like(input_ids) 
    opt_mask = torch.zeros_like(input_ids)
    opt_index = torch.zeros_like(input_ids)

    if question_length_limitation is not None:
        if len(question_ids) > question_length_limitation:
            question_ids = question_ids[:question_length_limitation]
    question_ids = [sep] + question_ids + [sep]
    question_length = len(question_ids)
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

    question_mask[0,1:question_length-1] = 1
    question_index[0, 1:question_length-1] = \
        torch.from_numpy(np.array(in_question_index[:question_length-2]))
    tags[0,  1:question_length-1] = torch.from_numpy(np.array(question_tags[:question_length-2]))

    table_mask[0, question_length:question_length + table_length] = 1
    table_index[0, question_length:question_length + table_length] = \
        torch.from_numpy(np.array(in_table_cell_index[:table_length]))
    tags[0, question_length:question_length + table_length] = torch.from_numpy(np.array(table_tags[:table_length]))
    if paragraph_length > 1:
        paragraph_mask[0, question_length + table_length + 1:question_length + table_length + paragraph_length] = 1
        paragraph_index[0, question_length + table_length + 1:question_length + table_length + paragraph_length] = \
            torch.from_numpy(np.array(in_paragraph_index[:paragraph_length - 1]))
        tags[0, question_length + table_length + 1:question_length + table_length + paragraph_length] = \
            torch.from_numpy(np.array(paragraph_tags[:paragraph_length - 1]))


    opt_mask[0,question_length + table_length + paragraph_length + 1 : question_length + table_length + paragraph_length + num_ops+1] = 1
    #opt_index[0,question_length + table_length + paragraph_length + 1:question_length + table_length + paragraph_length + num_ops+1] = torch.from_numpy(np.array([1,2,3,4,5,6]))

    del in_table_cell_index
    del in_paragraph_index
    del in_question_index

    return input_ids, attention_mask, paragraph_mask, paragraph_index, \
           table_mask,  table_index, tags, input_segments,opt_mask,opt_index,question_mask,question_index



def combine_tags(tags1,tags2):
    cat_tags = torch.cat((tags1[0],tags2[0]),dim = 0)
    sum_tags = torch.sum(cat_tags,dim = 0)
    com_tags = torch.where(sum_tags > 0 , 1 , 0)
    whole_tags = torch.zeros([1, 512])
    whole_tags[0] = com_tags
    return whole_tags
"""
instance format:
input_ids: np.array[1, 512]. The input ids.
attention_mask: np.array[1, 512]. The attention_mask to denote whether a id is real or padded.
token_type_ids: np.array[1, 512, 3]. 
    The special tokens needed by tapas within following orders: segment_ids, column_ids, row_ids.
tags_label: np.array[1, 512]. The tag ground truth indicate whether this id is part of the answer.
paragraph_mask: np.array[1, 512]. 1 for ids belongs to paragraph, 0 for others
paragraph_word_piece_mask: np.array[1, 512]. 0 for sub-token, 1 for non-sub-token or the start of sub-token
paragraph_number_value: np.array[1, 512]. nan for no-numerical words and number for numerical words extract from current word. i.e.: $460 -> 460
table_number_value: np.array[1, max_num_columns*max_num_rows]. Definition is the same as above.
paragraph_number_mask: np.array[1, 512]. 0 for non numerical token and 1 for numerical token.
table_number_mask: np.array[1, max_num_columns*max_num_rows]. 0 for non numerical token and 1 for numerical token.
paragraph_index: np.array[1, 512], used to apply token-lv reduce-mean after getting sequence_output
number_order_label: int. The operator calculating order.
operator_label:  int. The operator ground truth.
scale_label: int. The scale ground truth.
answer: str. The answer used to calculate metrics.
"""



class TagTaTQAReader(object):
    def __init__(self, tokenizer,
                 passage_length_limit: int = None, question_length_limit: int = None, sep="<s>", op_mode:int=8,
                 ablation_mode:int=0,num_ari_ops:int=6):
        self.max_pieces = 512
        self.tokenizer = tokenizer
        self.passage_length_limit = passage_length_limit
        self.question_length_limit = question_length_limit
        self.sep = self.tokenizer._convert_token_to_id(sep)
        self.opt = self.tokenizer.encode("<OPT>")[1]
        self.num_ops = num_ari_ops
        tokens = self.tokenizer._tokenize("Feb 2 Nov")
        self.skip_count = 0
        self.op_skip = 0
        self.ari_skip = 0
        self.op_mode=op_mode
        self.ari_ops = ARITHMETIC_CLASSES_
        if ablation_mode == 0:
            self.OPERATOR_CLASSES=OPERATOR_CLASSES_
        elif ablation_mode == 1:
            self.OPERATOR_CLASSES=get_op_1(op_mode)
        elif ablation_mode == 2:
            self.OPERATOR_CLASSES=get_op_2(op_mode)
        else:
            self.OPERATOR_CLASSES=get_op_3(op_mode)

    def _make_instance(self, input_ids, attention_mask, token_type_ids, paragraph_mask, table_mask,
                       paragraph_number_value, table_cell_number_value, paragraph_index, table_cell_index,
                        tags_ground_truth, operator_ground_truth, scale_ground_truth,paragraph_tokens, 
                        table_cell_tokens, answer_dict, question_id, ari_ops,opt_mask,opt_index,opt_labels,ari_labels,order_labels,
                       question_mask,question_index,question_number_value,question_tokens):

        if ari_ops != None:
            ari_ops_padding = self.num_ops - len(ari_ops)
            if ari_ops_padding > 0:
               ari_ops += [0]*ari_ops_padding
            if ari_ops == [0] * self.num_ops:
                print("no ari ops")
                ari_ops = [-100] * self.num_ops
            else:
                get0 = False
                for i in range(self.num_ops):
                    if get0 == True:
                        ari_ops[i] = -100
                    if ari_ops[i] == 0:
                        if get0 == False:
                            get0 = True
        else:
            ari_ops = [-100] * self.num_ops

        if ari_ops == [-100] * self.num_ops:
            opt_labels[0,:,:] = -100
            ari_labels[0,:,:] = -100
            order_labels[:] = -100
            number_indexes = []
            ari_sel_labels = []
            if ari_ops[1] == 0:
                opt_labels[0,:,:] = -100
        else:
            for i in range(1,self.num_ops-1):
                for j in range(i):
                    opt_labels[0,i,j] = -100
                if ari_ops[i] == 0:
                    ari_labels[0,i:,:] = -100
                    opt_labels[0,i-1:,:] = -100
                    opt_labels[0,:,i-1:] = -100

            number_indexes = []
            cur_indexes = []
            cur = 0
            selected_indexes = torch.nonzero(tags_ground_truth[0]).squeeze(-1)
            
            if len(selected_indexes) == 0:
                print("no number")
                return None
            for sel in selected_indexes:
                sel = int(sel)
                if int(question_index[0,sel]) != 0:
                    if int(question_index[0,sel]) == cur or cur_indexes == []:
                        if cur_indexes == []:
                            cur = int(question_index[0,sel])
                        cur_indexes.append(sel)
                    else:
                        cur = int(question_index[0,sel])
                        number_indexes.append(cur_indexes)
                        cur_indexes = [sel]
                elif int(table_cell_index[0,sel]) != 0:
                    if int(table_cell_index[0,sel]) == cur or cur_indexes == []:
                        if cur_indexes == []:
                            cur = int(table_cell_index[0,sel])
                        cur_indexes.append(sel)
                    else:
                        cur = int(table_cell_index[0,sel])
                        number_indexes.append(cur_indexes)
                        cur_indexes = [sel]
                else:
                    if int(paragraph_index[0,sel]) == cur or cur_indexes == []:
                        if cur_indexes == []:
                            cur = int(paragraph_index[0,sel])
                        cur_indexes.append(sel)
                    else:
                        cur = int(paragraph_index[0,sel])
                        number_indexes.append(cur_indexes)
                        cur_indexes = [sel]
            number_indexes.append(cur_indexes)
            distinct_si = []
            for  i , ni in enumerate(number_indexes):
                distinct_si.append(ni[0])
                p = 10 - len(ni)
                if  p > 0:
                    number_indexes[i] += [0] * p
                else:
                    print("long number")
                    if int(question_index[0,ni[0]]) != 0 :
                        print(question_number_value[int(question_index[0,ni[0]]) - 1])
                    elif int(table_cell_index[0,ni[0]]) != 0 :
                        print(table_cell_number_value[int(table_cell_index[0,ni[0]]) - 1])
                    elif int(paragraph_index[0,ni[0]]) != 0:
                        print(paragraph_number_value[int(paragraph_index[0,ni[0]]) - 1])
                    else:
                        print("extract err")

            ari_sel_labels = ari_labels[0,:,distinct_si].transpose(0,1)
            if ari_sel_labels.shape[0] != len(number_indexes):
                print(ari_sel_labels)
                print(number_indexes)
                exit(0)

            for i in range(self.num_ops):
                if ari_ops[i] in [2,4]:
                    count = 0
                    for sl in ari_sel_labels:
                        if sl[i] == 1:
                            count += 1
                    if count != 2:
                        order_labels[i] = -100
                else:
                    order_labels[i] = -100
        opt_id = torch.nonzero(opt_mask == 1)[0,1]
        #print(ari_ops)
        
        return {
            "input_ids": np.array(input_ids),
            "attention_mask": np.array(attention_mask),
            "token_type_ids": np.array(token_type_ids),
            "paragraph_mask": np.array(paragraph_mask),
            "table_mask": np.array(table_mask),
            "question_mask" : np.array(question_mask),
            "paragraph_number_value": np.array(paragraph_number_value),
            "table_cell_number_value": np.array(table_cell_number_value),
            "question_number_value": np.array(question_number_value),
            "paragraph_index": np.array(paragraph_index),
            "table_cell_index": np.array(table_cell_index),
            "question_index": np.array(question_index),
            "tag_labels": np.array(tags_ground_truth),
            "operator_label": int(operator_ground_truth),
            "scale_label": int(scale_ground_truth),
            "paragraph_tokens": paragraph_tokens,
            "table_cell_tokens": table_cell_tokens,
            "question_tokens": question_tokens,
            "answer_dict": answer_dict,
            "question_id": question_id,
            "ari_ops" : torch.LongTensor(ari_ops),
            "opt_mask" : opt_id,
            "order_labels" : torch.LongTensor(order_labels),
            "ari_labels" : torch.LongTensor(np.array(ari_sel_labels)),
            "selected_indexes" : np.array(number_indexes),
            "opt_labels": torch.LongTensor(np.array(opt_labels)),
        }

    def _to_instance(self, question: str, table: List[List[str]], paragraphs: List[Dict], answer_from: str,
                     answer_type: str, answer:str, derivation: str, facts:list,  answer_mapping: Dict, scale: str, question_id:str):

        
        question_text = question.strip()
        table_cell_tokens, table_ids, table_tags, table_cell_number_value, table_cell_index = \
                            table_tokenize(table, self.tokenizer, answer_mapping, answer_type)
        paragraph_tokens, paragraph_ids, paragraph_tags, paragraph_word_piece_mask, paragraph_number_mask, \
                paragraph_number_value, paragraph_index= \
            paragraph_tokenize(question, paragraphs, self.tokenizer, answer_mapping, answer_type)

        question_tokens, question_ids, question_tags, question_word_piece_mask, question_number_mask, \
                question_number_value, question_index= \
            question_tokenize(question_text, self.tokenizer, answer_mapping, answer_type)

        order_labels = np.zeros(self.num_ops)
        if answer_type == "arithmetic":
            operator_class = self.OPERATOR_CLASSES["ARITHMETIC"]
            num_facts = facts_to_nums(facts)
            isavg = 0
            try:
               if _is_average(num_facts,answer):
                   ari_ops = [self.ari_ops['AVERAGE']]
                   ari_tags = {'table':[table_tags],'para':[paragraph_tags], 'question' : question_tags,'operation':[[0]*self.num_ops]}
                   isavg = 1
            except:
                isavg = 0
            if isavg == 0:
               dvt_split_suc = 0
               try:
                    ari_operations = infix_evaluator(derivation,True)
                    dvt_split_suc = 1
                    if len(ari_operations) > self.num_ops:
                        operator_class = None
                        dvt_split_suc = 0
               except:
                   print("derivation split err")
                   operator_class = None
               if dvt_split_suc == 1:
                   ari_ops = [self.ari_ops[i[0]] for i in ari_operations]
                   operands = [i[1:] for i in ari_operations]
                   ari_tags = {'table':[],'para':[],'operation':[]}

                   for i,opds in enumerate(operands):
                       temp_mapping,operand_one_mapping,operand_two_mapping = split_mapping(opds,answer_mapping,table,paragraphs)
                       if temp_mapping == None:
                           operator_class = None
                           break
                       else:
                           if ari_operations[i][0] in ['DIFF','DIVIDE']:
                              if "question" in  operand_one_mapping and "question" in operand_two_mapping:
                                  if operand_one_mapping["question"][0][0] > operand_two_mapping["question"][0][0]:
                                      order_labels[i] = 1
                                  elif operand_one_mapping["question"][0][1] > operand_two_mapping["question"][0][1]:
                                      order_labels[i] = 1
                              elif ("table" in operand_one_mapping or "paragraph" in operand_one_mapping) and "question" in operand_two_mapping:
                                    order_labels[i] = 1
                              elif "table" in operand_one_mapping and "table" in operand_two_mapping:
                                  if operand_one_mapping["table"][0][0] > operand_two_mapping["table"][0][0]:
                                      order_labels[i] = 1
                                  elif operand_one_mapping["table"][0][1] > operand_two_mapping["table"][0][1]:
                                      order_labels[i] = 1
                              elif "paragraph" in operand_one_mapping and "table" in operand_two_mapping:
                                  order_labels[i] = 1
                              elif "paragraph" in operand_one_mapping and "paragraph" in operand_two_mapping:
                                  opd1_pid = list(operand_one_mapping["paragraph"].keys())[0]
                                  opd2_pid = list(operand_two_mapping["paragraph"].keys())[0]
                                  if int(opd1_pid) > int(opd2_pid):
                                      order_labels[i] = 1
                                  elif operand_one_mapping["paragraph"][opd1_pid][0][0] > operand_two_mapping["paragraph"][opd2_pid][0][0]:
                                      order_labels[i] = 1

                              _,_,op1_table_tags,_,_ = table_tokenize(table,self.tokenizer,operand_one_mapping,answer_type)
                              _,_,op1_question_tags,_,_,_,_ = question_tokenize(question,self.tokenizer,operand_one_mapping,answer_type)
                              _,_,op1_para_tags,_,_,_,_ = paragraph_tokenize(question, paragraphs, self.tokenizer, operand_one_mapping, answer_type)
                              _,_,op2_table_tags,_,_ = table_tokenize(table,self.tokenizer,operand_two_mapping,answer_type)
                              _,_,op2_para_tags,_,_,_,_ = paragraph_tokenize(question, paragraphs, self.tokenizer, operand_two_mapping, answer_type)
                              _,_,op2_question_tags,_,_,_,_ = question_tokenize(question,self.tokenizer,operand_two_mapping,answer_type)
                              ari_tags['table'].append({"operand1":op1_table_tags,"operand2":op2_table_tags})
                              ari_tags['para'].append({"operand1":op1_para_tags,"operand2":op2_para_tags})
                              ari_tags['question'].append({"operand1":op1_question_tags,"operand2":op2_question_tags})
                              op1_tags = [0] * self.num_ops
                              op2_tags = [0] * self.num_ops
                              if 'operator' in operand_one_mapping:
                                  for i in operand_one_mapping['operator']:
                                      op1_tags[i] = 1
                              if 'operator' in operand_two_mapping:
                                  for i in operand_two_mapping['operator']:
                                      op2_tags[i] = 1
                              ari_tags['operation'].append({"operand1":op1_tags,"operand2":op2_tags})

                           else:
                              _,_,temp_table_tags,_,_ = table_tokenize(table,self.tokenizer,temp_mapping,answer_type)
                              _,_,temp_para_tags,_,_,_,_ = paragraph_tokenize(question, paragraphs, self.tokenizer, temp_mapping, answer_type)
                              _,_,temp_question_tags,_,_,_,_ = question_tokenize(question, self.tokenizer, temp_mapping, answer_type)
                              ari_tags['table'].append(temp_table_tags)
                              ari_tags['para'].append(temp_para_tags)
                              ari_tags['question'].append(temp_question_tags)
                              temp_op_tags = [0] * self.num_ops
                              if 'operator' in temp_mapping:
                                  for i in temp_mapping['operator']:
                                      temp_op_tags[i] = 1
                              ari_tags['operation'].append(temp_op_tags)
        elif answer_type == "count":
            operator_class = self.OPERATOR_CLASSES["COUNT"]
            #ari_ops = [self.ari_ops['COUNT']]
            ari_ops = None
            ari_tags = None
        else:
            operator_class = get_operator_class(derivation, answer_type, facts, answer,
                                            answer_mapping, scale, self.OPERATOR_CLASSES)
            ari_ops = None
            ari_tags = None

        scale_class = SCALE.index(scale)
        if operator_class is None:
            self.skip_count += 1
            if answer_type == "arithmetic":
                self.ari_skip += 1
            else:
                self.op_skip += 1
            return None

        for i in range(len(table)):
            for j in range(len(table[i])):
                if table[i][j] == '' or table[i][j] == 'N/A' or table[i][j] == 'n/a':
                    table[i][j] = "NONE"
        table = pd.DataFrame(table, dtype=np.str)
        column_relation = {}
        for column_name in table.columns.values.tolist():
            column_relation[column_name] = str(column_name)
        table.rename(columns=column_relation, inplace=True)

        #question_ids = question_tokenizer(question_text, self.tokenizer)

        input_ids, attention_mask, paragraph_mask,  paragraph_index, \
        table_mask, table_index, tags, token_type_ids , opt_mask,opt_index,ari_round_labels,question_mask,question_index = \
            _concat(question_ids, question_tags, question_index,
                    table_ids, table_tags, table_cell_index, 
                    paragraph_ids, paragraph_tags, paragraph_index,
                    self.sep,self.opt,self.question_length_limit,
                    self.passage_length_limit, self.max_pieces,self.num_ops,ari_tags)


        opt_labels = torch.zeros(1,self.num_ops - 1 , self.num_ops-1)
        #whole_tags = combine_tags(ari_round_tags,opd_two_tags)
        if answer_type == "arithmetic":
            #tags = whole_tags
            ari_round_labels = torch.where(tags > 0 ,ari_round_labels,-100)
            if len(ari_ops) >= 2:
                #print(ari_tags["operation"])
                for i in range(1 , len(ari_ops)):
                    opt_tags = ari_tags["operation"][i]
                    if isinstance(opt_tags , dict):
                        opd1_opt_tags = opt_tags["operand1"]
                        for j in range(self.num_ops-1):
                            if opd1_opt_tags[j] == 1:
                                opt_labels[0,j,i-1] = 1
                        opd2_opt_tags = opt_tags["operand2"]
                        for j in range(self.num_ops-1):
                            if opd2_opt_tags[j] == 1:
                                opt_labels[0,j,i-1] = 2
                    else:
                        for j in range(self.num_ops-1):
                            if opt_tags[j] == 1:
                                opt_labels[0,j,i-1] = 1
                #print(opt_labels)
                #print(torch.nonzero(ari_round_labels == 2))
        answer_dict = {"answer_type": answer_type, "answer": answer, "scale": scale, "answer_from": answer_from}
        return self._make_instance(input_ids, attention_mask, token_type_ids, paragraph_mask, table_mask,
                    paragraph_number_value, table_cell_number_value, paragraph_index, table_index, tags, operator_class, scale_class,
                    paragraph_tokens, table_cell_tokens, answer_dict, question_id,ari_ops,opt_mask,opt_index,opt_labels,ari_round_labels,order_labels,
                    question_mask,question_index,question_number_value,question_tokens)


    def _read(self, file_path: str):
        print("Reading train dataset file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)
        instances = []
        key_error_count = 0
        index_error_count = 0
        assert_error_count = 0
        reading_cnt = 1

        for one in tqdm(dataset):
            table = one['table']['table']
            paragraphs = one['paragraphs']
            questions = one['questions']
            #print('')
            #print("***** Reading *****", reading_cnt)
            reading_cnt += 1
            for question_answer in questions:
                try:
                    is_counter = question_answer["counterfactual"] # is counterfactual? 0 or 1
                    question = question_answer["question"]
                    answer_type = question_answer["answer_type"]
                    answer = question_answer["answer"]
                    answer_from = question_answer["answer_from"]
                    scale = question_answer["scale"]
                    if is_counter:
                        question_if_part = question_answer["question_if_part"]
                        if answer_type == 'arithmetic': # counterfactual original questions
                            original_answer_mapping = question_answer["original_answer_mapping"] # use original mapping to find original operands
                            counter_answer_mapping = question_answer["mapping"] # dummy, not used
                            facts = question_answer["counter_facts"] if "counter_facts" in question_answer else question_answer["facts"] # for deciding operator
                            counter_derivation = question_answer["derivation"] # for deciding operator
                            original_derivation = question_answer["original_derivation"] # not used
                            if_mapping = question_answer["if_mapping"] if "if_mapping" in question_answer else {}
                            if_operator = question_answer["if_op"] if "if_op" in question_answer else 'NONE'
                        else: # original arithmetic questions
                            original_answer_mapping = question_answer["mapping"] # only using the corresponding mapping and derivation
                            counter_answer_mapping = question_answer["mapping"]
                            facts = question_answer["facts"]
                            counter_derivation = question_answer["derivation"]
                            original_derivation = question_answer["derivation"]
                            if_operator = 'NONE'
                            if_mapping = {}
                    else:
                        question_if_part = ""
                        counter_derivation = question_answer["derivation"]
                        original_derivation = question_answer["derivation"]
                        if_mapping = {}
                        if_operator = 'NONE'
                        original_answer_mapping = question_answer["mapping"]
                        counter_answer_mapping = question_answer["mapping"]
                        facts = question_answer["facts"]
                    
                    instance = self._to_instance(question, table, paragraphs, answer_from,
                                    answer_type, answer, counter_derivation,  facts,
                                    counter_answer_mapping, scale, question_answer["uid"])
                    if instance is not None:
                        instances.append(instance)
                except RuntimeError as e :
                    pass
                    #print(f"run time error:{e}" )
                    #print(question_answer["uid"])
                #except KeyError:
                #    key_error_count += 1
                #    print(question_answer["uid"])
                #    print("KeyError. Total Error Count: {}".format(key_error_count))
                #except IndexError:
                #    index_error_count += 1
                #    print(question_answer["uid"])
                #    print("IndexError. Total Error Count: {}".format(index_error_count))
                #except AssertionError:
                #    assert_error_count += 1
                #    print(question_answer["uid"])
                #    print("AssertError. Total Error Count: {}".format(assert_error_count))
        print('total instance numbers', len(instances))
        return instances

class TagTaTQATestReader(object):
    def __init__(self, tokenizer,
                 passage_length_limit: int = None, question_length_limit: int = None, sep="<s>",
                 ablation_mode=0, op_mode=0,num_ari_ops=6,mode ="dev"):
        self.max_pieces = 512
        self.tokenizer = tokenizer
        self.passage_length_limit = passage_length_limit
        self.question_length_limit = question_length_limit
        self.sep = self.tokenizer._convert_token_to_id(sep)
        self.opt = self.tokenizer.encode("<OPT>")[1]
        tokens = self.tokenizer._tokenize("Feb 2 Nov")
        self.skip_count = 0
        self.op_skip = 0
        self.mode = mode
        self.ari_skip = 0
        self.ari_ops = ARITHMETIC_CLASSES_
        self.OPERATOR_CLASSES=OPERATOR_CLASSES_
        self.num_ops = num_ari_ops
        self.op_count = {"Span-in-text":0, "Cell-in-table":0, "Spans":0, "Arithmetic":0, "Count":0}
        self.scale_count = {"":0, "thousand":0, "million":0, "billion":0, "percent":0}

    def _make_instance(self, input_ids, attention_mask, token_type_ids, paragraph_mask, table_mask,
                        paragraph_number_value, table_cell_number_value, paragraph_index, table_cell_index,
                        tags_ground_truth, paragraph_tokens, table_cell_tokens, answer_dict, question_id,opt_mask,opt_index,derivation,
                       question_mask,question_index,question_number_value,question_tokens):


        opt_id = torch.nonzero(opt_mask == 1)[0,1]
        return {
            "input_ids": np.array(input_ids),
            "attention_mask": np.array(attention_mask),
            "token_type_ids": np.array(token_type_ids),
            "paragraph_mask": np.array(paragraph_mask),
            "table_mask": np.array(table_mask),
            "paragraph_number_value": np.array(paragraph_number_value),
            "table_cell_number_value": np.array(table_cell_number_value),
            "question_number_value": np.array(question_number_value),
            "paragraph_index": np.array(paragraph_index),
            "table_cell_index": np.array(table_cell_index),
            "question_index": np.array(question_index),
            "tag_labels": np.array(tags_ground_truth),
            "paragraph_tokens": paragraph_tokens,
            "table_cell_tokens": table_cell_tokens,
            "question_tokens": question_tokens,
            "answer_dict": answer_dict,
            "question_id": question_id,
            "opt_mask":opt_id,
            "derivation":derivation,
            "question_mask":np.array(question_mask),
            
        }

    def summerize_op(self, derivation, answer_type, facts, answer, answer_mapping, scale,tv,pv,qv):
        truth_numbers = []
        order_labels = [-100]*self.num_ops
        if answer_type == "span":
            if "table" in answer_mapping.keys():
                self.op_count["Cell-in-table"] += 1
                return ["Cell-in-table"] +["ignore"]* (self.num_ops-1),truth_numbers,order_labels
            elif "paragraph" in answer_mapping.keys():
                self.op_count["Span-in-text"] += 1
                return ["Span-in-text"] +["ignore"]* (self.num_ops-1) , truth_numbers,order_labels
        elif answer_type == "multi-span":
            self.op_count["Spans"] += 1
            return ["Spans"]  +["ignore"]* (self.num_ops-1),truth_numbers,order_labels
        elif answer_type == "count":
            self.op_count["Count"] += 1
            return ["Count"] +["ignore"] * (self.num_ops-1),truth_numbers,order_labels
        elif answer_type == "arithmetic":
            num_facts = facts_to_nums(facts)
            if not is_number(str(answer)):
                return "",truth_numbers
            else:
                self.op_count["Arithmetic"] += 1
                isavg = 0
                try:
                   if _is_average(num_facts,answer):
                       operator_classes = ["Average"]+["Stop"] +["ignore"]* (self.num_ops-2)
                       isavg = 1
                except:
                   isavg = 0
                if isavg == 0:
                    dvt_split_suc = 0
                    try:
                       ari_operations = infix_evaluator(derivation,True)
                       #print(len(ari_operations))
                       for ari in ari_operations:
                           for num in ari[1:]:
                               if not isinstance(num,str):
                                   if num not in truth_numbers:
                                       truth_numbers.append(num)
                       dvt_split_suc = 1
                       if len(ari_operations) > self.num_ops:
                           operator_classes = None
                           dvt_split_suc = 0
                    except:
                       print(derivation)
                       operator_classes = None
                    if dvt_split_suc == 1:
                        operator_classes = ["ignore"]*self.num_ops
                        for i,ari in enumerate(ari_operations):
                            if ari[0] == "SUM":
                                operator_classes[i] = "Sum"
                            if ari[0] == "DIFF":
                                operator_classes[i] = "Difference"
                                opd1 = -100
                                opd2 = -100
                                ql = len(qv)
                                tl = len(tv)
                                pl = len(pv)
                                for o in range(ql+tl+pl):
                                    if isinstance(ari[1],str) or isinstance(ari[2],str):
                                        break
                                    if o < ql:
                                        if qv[o] == ari[1]:
                                            opd1 = o
                                        if qv[o] == ari[2]:
                                            opd2 = o
                                    elif o < tl:
                                        if tv[o] == ari[1]:
                                            opd1 = o
                                        if tv[o] == ari[2]:
                                            opd2 = o
                                    else:
                                        if pv[o-tl] == ari[1]:
                                            opd1 = o
                                        if pv[o-tl] == ari[2]:
                                            opd2 = o
                                if opd1 == -100 or opd2 == -100:
                                    print("order fail")
                                    order_labels[i] = -100
                                else:
                                    if opd1 <= opd2:
                                        order_labels[i] = 0
                                    else:
                                        order_labels[i] = 1
                            if ari[0] == "TIMES":
                                operator_classes[i] = "Multiplication"
                            if ari[0] == "DIVIDE":
                                operator_classes[i] = "Division"
                                opd1 = -100
                                opd2 = -100
                                tl = len(tv)
                                pl = len(pv)
                                for o in range(tl+pl):
                                    if isinstance(ari[1],str) or isinstance(ari[2],str):
                                        break
                                    if o < ql:
                                        if qv[o] == ari[1]:
                                            opd1 = o
                                        if qv[o] == ari[2]:
                                            opd2 = o
                                    elif o < tl:
                                        if tv[o] == ari[1]:
                                            opd1 = o
                                        if tv[o] == ari[2]:
                                            opd2 = o
                                    else:
                                        if pv[o-tl] == ari[1]:
                                            opd1 = o
                                        if pv[o-tl] == ari[2]:
                                            opd2 = o
                                if opd1 == -100 or opd2 == -100:
                                    print("order fail")
                                    order_labels[i] = -100
                                else:
                                    if opd1 <= opd2:
                                        order_labels[i] = 0
                                    else:
                                        order_labels[i] = 1
                            if ari[0] == "AVERAGE":
                                operator_classes[i] = "Average"
                            if ari[0] == "INC":
                                 operator_classes[i] = "ratio increasing"
                            if ari[0] == "DEC":
                                 operator_classes[i] = "ratio decreasing"
                            j = i
                        #print(j)
                        if j < self.num_ops - 1:
                            operator_classes[j+1] = "Stop"
                    else:
                       operator_classes = None

            return operator_classes,truth_numbers,order_labels

    def _to_test_instance(self, question: str, table: List[List[str]], paragraphs: List[Dict], answer_from: str,
                     answer_type: str, answer:str, answer_mapping, scale: str, question_id:str, derivation, facts):
        question_text = question.strip()

        '''
        if self.mode == "dev":
            gold_ops,truth_numbers = self.summerize_op(derivation, answer_type, facts, answer, answer_mapping, scale)
            if gold_ops is None:
                gold_ops = ["ignore"] *self.num_ops
        '''

        table_cell_tokens, table_ids, table_tags, table_cell_number_value, table_cell_index = \
            table_test_tokenize(table, self.tokenizer, answer_mapping, answer_type)

        paragraph_tokens, paragraph_ids, paragraph_tags, paragraph_word_piece_mask, paragraph_number_mask, \
                paragraph_number_value, paragraph_index, paragraph_mapping_content = \
            paragraph_test_tokenize(question, paragraphs, self.tokenizer, answer_mapping, answer_type)
        
        question_tokens, question_ids, question_tags, question_word_piece_mask, question_number_mask, \
                question_number_value, question_index= \
            question_tokenize(question_text, self.tokenizer, answer_mapping, answer_type)

        if self.mode == "dev":
            gold_ops,truth_numbers,order_labels = self.summerize_op(derivation, answer_type, facts, answer, answer_mapping, scale,
                                                                    table_cell_number_value,paragraph_number_value,question_number_value)
            if gold_ops is None:
                gold_ops = ["ignore"] *self.num_ops
        #question_ids = question_tokenizer(question_text, self.tokenizer)

        input_ids, attention_mask, paragraph_mask,  paragraph_index, \
        table_mask, table_index, tags, token_type_ids ,opt_mask,opt_index , question_mask,question_index= \
            _test_concat(question_ids, question_tags , question_index,
                    table_ids, table_tags, table_cell_index,
                    paragraph_ids, paragraph_tags, paragraph_index,
                    self.sep,self.opt, self.question_length_limit,
                    self.passage_length_limit, self.max_pieces,self.num_ops)

        if self.mode == "test":
            gold_ops= None
            scale = None
            order_labels = None
        else:
            self.scale_count[scale] += 1
        answer_dict = {"answer_type": answer_type, "answer": answer, "scale": scale, "answer_from": answer_from,
                "gold_ops": gold_ops, "gold_scale":scale,"order_labels":order_labels}

        return self._make_instance(input_ids, attention_mask, token_type_ids, paragraph_mask, table_mask,
            paragraph_number_value, table_cell_number_value, paragraph_index, table_index, tags, paragraph_tokens,
            table_cell_tokens, answer_dict, question_id,opt_mask,opt_index,derivation,question_mask,question_index,question_number_value,question_tokens)


    def _read(self, file_path: str):
        print("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)
            dataset_file.close()
        #f = open("3round_dev.json",'w')
        print("Reading the tatqa dataset")
        instances = []
        maxround_instances = []
        index_error_count = 0
        assert_error_count = 0
        for one in tqdm(dataset):
            table = one['table']['table']
            paragraphs = one['paragraphs']
            questions = one['questions']
            for question_answer in questions:
                question = question_answer["question"].strip()
                if self.mode == "dev":
                    answer_type = question_answer["answer_type"]
                    answer = question_answer["answer"]
                    answer_from = question_answer["answer_from"]
                    answer_mapping = question_answer["mapping"]
                    scale = question_answer["scale"]
                    derivation = question_answer['derivation']
                    facts = question_answer['facts']
                else:
                    answer_type = None
                    answer = None
                    answer_from = None
                    answer_mapping = None
                    scale = None
                    derivation = None
                    facts = None
                instance = self._to_test_instance(question, table, paragraphs, answer_from,answer_type, answer,
                                                  answer_mapping, scale, question_answer["uid"], derivation, facts)
                #if instance is not None and "ignore" not in instance["answer_dict"]["gold_ops"][:3] and "Stop" not in instance["answer_dict"]["gold_ops"][:3]:
                if instance is not None:
                    instances.append(instance)
                    #if "Stop" not in instance["answer_dict"]["gold_ops"] and "ignore" not in instance["answer_dict"]["gold_ops"]:
                    if instance["answer_dict"]["gold_ops"][3] == "Stop":
                       maxround_instances.append(instance)
        print(self.op_count)
        print(self.scale_count)
        self.op_count = {"Span-in-text": 0, "Cell-in-table": 0, "Spans": 0,"Arithmetic":0,"Count":0}
        self.scale_count = {"": 0, "thousand": 0, "million": 0, "billion": 0, "percent": 0}
        return instances,maxround_instances

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
    vector_shape = values.size()[len(index.indices.size()) :]  # torch.Size object
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
