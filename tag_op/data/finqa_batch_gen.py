import os
import pickle
import torch
import random
import numpy as np
class TaTQABatchGen(object):
    def __init__(self, args, data_mode,num_ops ,encoder='roberta'):
        dpath =  f"tagop_{encoder}_cached_{data_mode}.pkl"
        self.is_train = data_mode == "train"
        self.args = args
        self.num_ops = num_ops
        with open(os.path.join(args.data_dir, dpath), 'rb') as f:
            print("Load data from {}.".format(dpath))
            data = pickle.load(f)

        all_data = []
        for item in data:
            input_ids = torch.from_numpy(item["input_ids"])
            attention_mask = torch.from_numpy(item["attention_mask"])
            token_type_ids = torch.from_numpy(item["token_type_ids"])
            paragraph_mask = torch.from_numpy(item["paragraph_mask"])
            table_mask = torch.from_numpy(item["table_mask"])
            paragraph_numbers = item["paragraph_number_value"]
            table_cell_numbers = item["table_cell_number_value"]
            col_index = item["col_index"]
            paragraph_index = torch.from_numpy(item["paragraph_index"])
            table_cell_index = torch.from_numpy(item["table_cell_index"])
            tag_labels = torch.from_numpy(item["tag_labels"])
            task_labels = torch.tensor(item["operator_label"])
            gold_answers = item["answer_dict"]
            paragraph_tokens = item["paragraph_tokens"]
            table_cell_tokens = item["table_cell_tokens"]
            question_id = item["question_id"]
            opt_mask = item["opt_mask"]
            ari_ops = item["ari_ops"]
            opt_labels = item["opt_labels"]
            ari_labels = item["ari_labels"]
            selected_indexes = item["selected_indexes"]
            order_labels = item["order_labels"]
            
            all_data.append((input_ids, attention_mask, token_type_ids, paragraph_mask, table_mask, paragraph_index,
                table_cell_index, tag_labels, task_labels,  gold_answers,
                paragraph_tokens, table_cell_tokens, paragraph_numbers, table_cell_numbers, col_index,
                question_id,ari_ops,opt_labels,ari_labels,opt_mask,order_labels,selected_indexes
                ))
        print("Load data size {}.".format(len(all_data)))
        self.data = TaTQABatchGen.make_batches(all_data, args.batch_size if self.is_train else args.eval_batch_size,
                                              self.is_train)
        self.offset = 0
        self.all_data = all_data

    @staticmethod
    def make_batches(data, batch_size=32, is_train=True):
        if is_train:
            random.shuffle(data)
        if is_train:
            return [
                data[i: i + batch_size] if i + batch_size < len(data) else data[i:] + data[
                                                                                      :i + batch_size - len(data)]
                for i in range(0, len(data), batch_size)]
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    def reset(self):
        if self.is_train:
            indices = list(range(len(self.data)))
            random.shuffle(indices)
            self.data = [self.data[i] for i in indices]
            for i in range(len(self.data)):
                random.shuffle(self.data[i])
        self.offset = 0

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        while self.offset < len(self):
            batch = self.data[self.offset]
            self.offset += 1

            input_ids_batch, attention_mask_batch, token_type_ids_batch, paragraph_mask_batch, table_mask_batch, \
            paragraph_index_batch, table_cell_index_batch, tag_labels_batch, task_labels_batch, \
            gold_answers_batch, paragraph_tokens_batch, table_cell_tokens_batch, paragraph_numbers_batch,\
            table_cell_numbers_batch, col_index_batch,question_ids_batch,  ari_ops_batch ,\
            opt_labels_batch , ari_labels_batch,opt_mask_batch,order_labels_batch , \
            selected_indexes_batch  = zip(*batch)

            bsz = len(batch)
            input_ids = torch.LongTensor(bsz, 512)
            attention_mask = torch.LongTensor(bsz, 512)
            token_type_ids = torch.LongTensor(bsz, 512).fill_(0)
            paragraph_mask = torch.LongTensor(bsz, 512)
            table_mask = torch.LongTensor(bsz, 512)
            paragraph_index = torch.LongTensor(bsz, 512)
            table_cell_index = torch.LongTensor(bsz, 512)
            tag_labels = torch.LongTensor(bsz, 512)
            task_labels = torch.LongTensor(bsz)

            ari_labels = torch.LongTensor([])
            selected_indexes = np.zeros([1,11])

            opt_mask = torch.LongTensor(bsz)
            ari_ops = torch.LongTensor(bsz,self.num_ops)

            opt_labels = torch.LongTensor(bsz,self.num_ops-1,self.num_ops-1)

            order_labels = torch.LongTensor(bsz,self.num_ops)

            paragraph_tokens = []
            table_cell_tokens = []
            gold_answers = []
            question_ids = []
            paragraph_numbers = []
            table_cell_numbers = []
            col_index = []
            for i in range(bsz):
                input_ids[i] = input_ids_batch[i]
                attention_mask[i] = attention_mask_batch[i]
                token_type_ids[i] = token_type_ids_batch[i]
                paragraph_mask[i] = paragraph_mask_batch[i]
                table_mask[i] = table_mask_batch[i]
                paragraph_index[i] = paragraph_index_batch[i]
                opt_mask[i] = opt_mask_batch[i]
                table_cell_index[i] = table_cell_index_batch[i]
                tag_labels[i] = tag_labels_batch[i]
                task_labels[i] = operator_labels_batch[i]
                ari_ops[i] = torch.LongTensor(ari_ops_batch[i])
                if len(selected_indexes_batch[i]) != 0:
                    ari_labels = torch.cat((ari_labels , ari_labels_batch[i]) , dim = 0)
                    num = selected_indexes_batch[i].shape[0]
                    sib = np.zeros([num,11])
                    for j in range(num):
                        sib[j,0] = i
                        try:
                            sib[j,1:] = selected_indexes_batch[i][j]
                        except:
                            print(selected_indexes_batch[i][j])
                            sib[j,1:] = selected_indexes_batch[i][j][:10]
                    selected_indexes = np.concatenate((selected_indexes , sib) , axis = 0)

                order_labels[i] = order_labels_batch[i]
                opt_labels[i] = opt_labels_batch[i]
                paragraph_tokens.append(paragraph_tokens_batch[i])
                table_cell_tokens.append(table_cell_tokens_batch[i])
                paragraph_numbers.append(paragraph_numbers_batch[i])
                table_cell_numbers.append(table_cell_numbers_batch[i])
                col_index.append(col_index_batch[i])
                gold_answers.append(gold_answers_batch[i])
                question_ids.append(question_ids_batch[i])

            out_batch = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids":token_type_ids,
                "paragraph_mask": paragraph_mask, "paragraph_index": paragraph_index, "tag_labels": tag_labels,
                "task_labels": task_labels, "paragraph_tokens": paragraph_tokens, 
                "table_cell_tokens": table_cell_tokens, "paragraph_numbers": paragraph_numbers,
                "table_cell_numbers": table_cell_numbers,"col_index":col_index, "gold_answers": gold_answers, "question_ids": question_ids,
                "table_mask": table_mask, "table_cell_index":table_cell_index, "ari_ops":ari_ops,
                "ari_labels":ari_labels,"opt_labels":opt_labels,"opt_mask":opt_mask,"order_labels":order_labels,
                "selected_indexes" : selected_indexes[1:]
            }

            if self.args.cuda:
                for k in out_batch.keys():
                    if isinstance(out_batch[k], torch.Tensor):
                        out_batch[k] = out_batch[k].cuda()

            yield  out_batch

class TaTQATestBatchGen(object):
    def __init__(self, args, data_mode,num_ops, encoder='roberta'):
        dpath =  f"tagop_{encoder}_cached_{data_mode}.pkl"
        self.is_train = data_mode == "train"
        self.args = args
        self.num_ops = num_ops
        print(os.path.join(args.test_data_dir, dpath))
        with open(os.path.join(args.test_data_dir, dpath), 'rb') as f:
            print("Load data from {}.".format(dpath))
            data = pickle.load(f)

        all_data = []

        if data_mode == "test":
            data = data[0]
        for item in data:
            input_ids = torch.from_numpy(item["input_ids"])
            attention_mask = torch.from_numpy(item["attention_mask"])
            token_type_ids = torch.from_numpy(item["token_type_ids"])
            paragraph_mask = torch.from_numpy(item["paragraph_mask"])
            table_mask = torch.from_numpy(item["table_mask"])
            paragraph_numbers = item["paragraph_number_value"]
            table_cell_numbers = item["table_cell_number_value"]
            paragraph_index = torch.from_numpy(item["paragraph_index"])
            table_cell_index = torch.from_numpy(item["table_cell_index"])
            tag_labels = torch.from_numpy(item["tag_labels"])
            gold_answers = item["answer_dict"]
            paragraph_tokens = item["paragraph_tokens"]
            table_cell_tokens = item["table_cell_tokens"]
            question_id = item["question_id"]
            derivation = item["derivation"]
            #ari_ops = item["ari_ops"]
            opt_mask = item["opt_mask"]

            question_mask = torch.from_numpy(item["question_mask"])
            #truth_numbers = item["truth_numbers"]
            #opt_index = item["opt_index"]


            all_data.append((input_ids, attention_mask, token_type_ids, paragraph_mask, table_mask, paragraph_index,
                             table_cell_index, tag_labels, gold_answers, paragraph_tokens, table_cell_tokens,
                             paragraph_numbers, table_cell_numbers, question_id,opt_mask,derivation,question_mask))
        print("Load data size {}.".format(len(all_data)))
        self.data = TaTQATestBatchGen.make_batches(all_data, args.batch_size if self.is_train else args.eval_batch_size,
                                               self.is_train)
        self.offset = 0

    @staticmethod
    def make_batches(data, batch_size=32, is_train=True):
        if is_train:
            random.shuffle(data)
        if is_train:
            return [
                data[i: i + batch_size] if i + batch_size < len(data) else data[i:] + data[
                                                                                      :i + batch_size - len(data)]
                for i in range(0, len(data), batch_size)]
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    def reset(self):
        if self.is_train:
            indices = list(range(len(self.data)))
            random.shuffle(indices)
            self.data = [self.data[i] for i in indices]
            for i in range(len(self.data)):
                random.shuffle(self.data[i])
        self.offset = 0

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        while self.offset < len(self):
            batch = self.data[self.offset]
            self.offset += 1
            input_ids_batch, attention_mask_batch, token_type_ids_batch, paragraph_mask_batch, table_mask_batch, \
            paragraph_index_batch, table_cell_index_batch, tag_labels_batch, gold_answers_batch, paragraph_tokens_batch, \
            table_cell_tokens_batch, paragraph_numbers_batch, table_cell_numbers_batch, question_ids_batch,opt_mask_batch,derivation_batch,question_mask_batch = zip(*batch)
            bsz = len(batch)
            input_ids = torch.LongTensor(bsz, 512)
            attention_mask = torch.LongTensor(bsz, 512)
            #token_type_ids = torch.LongTensor(bsz, 512).fill_(0)
            token_type_ids = torch.LongTensor(bsz, 512,7)
            paragraph_mask = torch.LongTensor(bsz, 512)
            table_mask = torch.LongTensor(bsz, 512)
            paragraph_index = torch.LongTensor(bsz, 512)
            table_cell_index = torch.LongTensor(bsz, 512)
            tag_labels = torch.LongTensor(bsz, 512)
            question_mask = torch.LongTensor(bsz,512)
            opt_mask = torch.LongTensor(bsz)
            #opt_index = torch.LongTensor(bsz,self.num_ops,1024)
            #ari_ops = torch.LongTensor(bsz,self.num_ops)


            paragraph_tokens = []

            table_cell_tokens = []
            gold_answers = []
            question_ids = []
            paragraph_numbers = []
            table_cell_numbers = []
            #truth_numbers = []

            derivation = []

            for i in range(bsz):
                input_ids[i] = input_ids_batch[i]
                attention_mask[i] = attention_mask_batch[i]
                token_type_ids[i] = token_type_ids_batch[i]
                paragraph_mask[i] = paragraph_mask_batch[i]
                table_mask[i] = table_mask_batch[i]
                paragraph_index[i] = paragraph_index_batch[i]
                table_cell_index[i] = table_cell_index_batch[i]
                tag_labels[i] = tag_labels_batch[i]
                paragraph_tokens.append(paragraph_tokens_batch[i])
                table_cell_tokens.append(table_cell_tokens_batch[i])
                paragraph_numbers.append(paragraph_numbers_batch[i])
                table_cell_numbers.append(table_cell_numbers_batch[i])
                gold_answers.append(gold_answers_batch[i])
                question_ids.append(question_ids_batch[i])
                derivation.append(derivation_batch[i])
                #truth_numbers.append(truth_numbers_batch[i])
                question_mask[i] = question_mask_batch[i]

                opt_mask[i] = opt_mask_batch[i]
                #opt_index[i] = opt_index_batch[i]
                #ari_ops[i] = ari_ops_batch[i]
            out_batch = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids,
                         "paragraph_mask": paragraph_mask, "paragraph_index": paragraph_index, "tag_labels": tag_labels,
                         "paragraph_tokens": paragraph_tokens, "table_cell_tokens": table_cell_tokens,
                         "paragraph_numbers": paragraph_numbers,
                         "table_cell_numbers": table_cell_numbers, "gold_answers": gold_answers, "question_ids": question_ids,
                         "table_mask": table_mask, "table_cell_index": table_cell_index,"opt_mask":opt_mask,"derivation":derivation,"question_mask":question_mask
                         # "paragraph_mapping_content": paragraph_mapping_content,
                         # "table_mapping_content": table_mapping_content,
                         }

            if self.args.cuda:
                for k in out_batch.keys():
                    if isinstance(out_batch[k], torch.Tensor):
                        out_batch[k] = out_batch[k].cuda()

            yield  out_batch
