import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
from tatqa_metric import TaTQAEmAndF1
from .tools.util import FFNLayer
from .tools import allennlp as util
from typing import Dict, List, Tuple
import numpy as np
from tag_op.data.file_utils import is_scatter_available
from tag_op.data.data_util import get_op_1, get_op_2, get_op_3, SCALE, OPERATOR_CLASSES_,ARITHMETIC_CLASSES_

np.set_printoptions(threshold=np.inf)
# soft dependency
if is_scatter_available():
    from torch_scatter import scatter
    from torch_scatter import scatter_max


ari_class_keys = list(ARITHMETIC_CLASSES_.keys())
def replace_opt(tensor,mask,ari_operator_ids,pred_ari_class,batch_size,num_ops):
    new_tensor = tensor.clone()
    for bsz in range(batch_size):
        for roud in range(num_ops):
            new_tensor[bsz,mask[bsz]+roud] = ari_operator_ids['<'+ari_class_keys[pred_ari_class[bsz,roud]]+'>']
        #mask[bsz] += 1
    return new_tensor



def get_continuous_tag_slots(paragraph_token_tag_prediction):
    tag_slots = []
    span_start = False
    for i in range(1, len(paragraph_token_tag_prediction)):
        if paragraph_token_tag_prediction[i] != 0 and not span_start:
            span_start = True
            start_index = i
        if paragraph_token_tag_prediction[i] == 0 and span_start:
            span_start = False
            tag_slots.append((start_index, i))
    if span_start:
        tag_slots.append((start_index, len(paragraph_token_tag_prediction)))
    return tag_slots


def get_span_tokens_from_paragraph(paragraph_token_tag_prediction, paragraph_tokens) -> List[str]:
    span_tokens = []
    span_start = False
    for i in range(1, min(len(paragraph_tokens) + 1, len(paragraph_token_tag_prediction))):
        if paragraph_token_tag_prediction[i] == 0:
            span_start = False
        if paragraph_token_tag_prediction[i] != 0:
            if not span_start:
                span_tokens.append([paragraph_tokens[i - 1]])
                span_start = True
            else:
                span_tokens[-1] += [paragraph_tokens[i - 1]]
    span_tokens = [" ".join(tokens) for tokens in span_tokens]
    return span_tokens


def get_span_tokens_from_table(table_cell_tag_prediction, table_cell_tokens) -> List[str]:
    span_tokens = []
    for i in range(1, len(table_cell_tag_prediction)):
        if table_cell_tag_prediction[i] != 0:
            span_tokens.append(str(table_cell_tokens[i-1]))
    return span_tokens


def get_single_span_tokens_from_paragraph(paragraph_token_tag_prediction,
                                          paragraph_token_tag_prediction_score,
                                          paragraph_tokens) -> List[str]:
    tag_slots = get_continuous_tag_slots(paragraph_token_tag_prediction)
    best_result = float("-inf")
    best_combine = []
    for tag_slot in tag_slots:
        current_result = np.mean(paragraph_token_tag_prediction_score[tag_slot[0]:tag_slot[1]])
        if current_result > best_result:
            best_result = current_result
            best_combine = tag_slot
    if not best_combine:
        return []
    else:
        return [" ".join(paragraph_tokens[best_combine[0] - 1: best_combine[1] - 1])]

def get_single_span_tokens_from_table(table_cell_tag_prediction,
                                      table_cell_tag_prediction_score,
                                      table_cell_tokens) -> List[str]:
    tagged_cell_index = [i for i in range(len(table_cell_tag_prediction)) if table_cell_tag_prediction[i] != 0]
    if not tagged_cell_index:
        return []
    tagged_cell_tag_prediction_score = \
        [table_cell_tag_prediction_score[i] for i in tagged_cell_index]
    best_result_index = tagged_cell_index[int(np.argmax(tagged_cell_tag_prediction_score))]
    return [str(table_cell_tokens[best_result_index-1])]

def get_numbers_from_reduce_sequence(sequence_reduce_tag_prediction, sequence_numbers):
    return [sequence_numbers[i - 1] for i in
            range(1, min(len(sequence_numbers) + 1, len(sequence_reduce_tag_prediction)))
            if sequence_reduce_tag_prediction[i] != 0 and np.isnan(sequence_numbers[i - 1]) != True]


def get_numbers_from_table(cell_tag_prediction, table_numbers):
    return [table_numbers[i] for i in range(len(cell_tag_prediction)) if cell_tag_prediction[i] != 0 and \
            np.isnan(table_numbers[i]) != True]

def get_number_index_from_reduce_sequence(sequence_reduce_tag_prediction, sequence_numbers):
    indexes = []
    numbers = []
    for i in range(1, min(len(sequence_numbers) + 1, len(sequence_reduce_tag_prediction))):
        if sequence_reduce_tag_prediction[i] != 0 and np.isnan(sequence_numbers[i - 1]) != True:
            indexes.append(i)
            numbers.append(sequence_numbers[i - 1])
    return indexes , numbers

class TagopModel(nn.Module):
    def __init__(self,
                 encoder,
                 config,
                 bsz,
                 operator_classes: int,
                 ari_classes:int,
                 scale_classes: int,
                 num_ops : int,
                 ari_operator_ids : dict,
                 operator_criterion: nn.CrossEntropyLoss = None,
                 ari_criterion: nn.CrossEntropyLoss = None,
                 opt_criterion: nn.CrossEntropyLoss = None,
                 ari_operator_criterion: nn.CrossEntropyLoss = None,
                 scale_criterion: nn.CrossEntropyLoss = None,
                 hidden_size: int = None,
                 dropout_prob: float = None,
                 arithmetic_op_index: List = None,
                 op_mode: int = None,
                 ablation_mode: int = None,
                 ):
        super(TagopModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.operator_classes = operator_classes
        self.ari_classes = ari_classes
        self.scale_classes = scale_classes
        self.num_ops = num_ops
        if hidden_size is None:
            hidden_size = self.config.hidden_size

        self.hidden_size = hidden_size
        if dropout_prob is None:
            dropout_prob = self.config.hidden_dropout_prob
        # operator predictor
        self.operator_predictor = FFNLayer(hidden_size, hidden_size, operator_classes, dropout_prob)

        self.ari_predictor = FFNLayer(hidden_size, hidden_size, ari_classes, dropout_prob)
        # scale predictor
        self.scale_predictor = FFNLayer(3 * hidden_size, hidden_size, scale_classes, dropout_prob)
        # tag predictor: two-class classification
        self.tag_predictor = FFNLayer(hidden_size, hidden_size, 2, dropout_prob)

        self.operand_predictor = FFNLayer(2*hidden_size, hidden_size, 3, dropout_prob)
        self.opt_predictor = FFNLayer(2*hidden_size, hidden_size, 3, dropout_prob)
        # order predictor: three-class classification
        #self.order_predictor = FFNLayer(hidden_size, hidden_size, 2, dropout_prob)
        # criterion for operator/scale loss calculation
        self.operator_criterion = operator_criterion
        self.scale_criterion = scale_criterion
        self.ari_criterion = ari_criterion
        self.opt_criterion = opt_criterion
        self.ari_operator_criterion = ari_operator_criterion
        # NLLLoss for tag_prediction
        self.NLLLoss = nn.NLLLoss(reduction="sum")
        # tapas config
        self.config = config

        self.arithmetic_op_index = arithmetic_op_index

        self.ARI_CLASSES = ARITHMETIC_CLASSES_
        self.ari_operator_ids = ari_operator_ids
        if ablation_mode == 0:
            self.OPERATOR_CLASSES = OPERATOR_CLASSES_
        elif ablation_mode == 1:
            self.OPERATOR_CLASSES = get_op_1(op_mode)
        elif ablation_mode == 2:
            self.OPERATOR_CLASSES = get_op_2(op_mode)
        else:
            self.OPERATOR_CLASSES = get_op_3(op_mode)
        self._metrics = TaTQAEmAndF1()

    """
    :parameter
    input_ids, shape:[bsz, 512] split_tokens' ids, 0 for padded token.
    attention_mask, shape:[bsz, 512] 0 for padded token and 1 for others
    token_type_ids, shape[bsz, 512, 3].
    # row_ids and column_ids are non-zero for table-contents and 0 for others, including headers.
        segment_ids[:, :, 0]: 1 for table and 0 for others
        column_ids[:, :, 1]: indicate to which column of the table a token belongs (starting from 1). Is 0 for all question
      tokens, special tokens and padding.
        row_ids[:, :, 2]: indicate to which row of the table a token belongs (starting from 1). Is 0 for all question tokens,
      special tokens and padding. Tokens of column headers are also 0.
    paragraph_mask, shape[bsz, 512] 1 for paragraph_tokens and 0 for others
    paragraph_index, shape[bsz, 512] 0 for non-paragraph tokens and index starting from 1 for paragraph tokens
    tag_labels: [bsz, 512] 1 for tokens in the answer and 0 for others
    operator_labels: [bsz, 8]
    scale_labels: [bsz, 8]
    number_order_labels: [bsz, 2]
    paragraph_tokens: [bsz, text_len], white-space split tokens
    tables: [bsz,] raw tables in DataFrame.
    paragraph_numbers: [bsz, text_len], corresponding number extracted from tokens, nan for non-number token
    table_numbers: [bsz, table_size], corresponding number extracted from table cells, nan for non-number cell. Shape is the same as flattened table.
    """

    def forward(self,
                input_ids: torch.LongTensor,
                attention_mask: torch.LongTensor,
                token_type_ids: torch.LongTensor,
                paragraph_mask: torch.LongTensor,
                paragraph_index: torch.LongTensor,
                opt_mask : torch.LongTensor,
                #opt_index : torch.LongTensor,
                tag_labels: torch.LongTensor,
                #ari_tags:torch.LongTensor,
                #opd_two_tags:torch.LongTensor,
                operator_labels: torch.LongTensor,
                ari_ops:torch.LongTensor,
                #pos_opt_labels : torch.LongTensor,
                #neg_opt_labels : torch.LongTensor,

                ari_labels : torch.LongTensor,
                opt_labels : torch.LongTensor,
                scale_labels: torch.LongTensor,
                gold_answers: str,
                paragraph_tokens: List[List[str]],
                paragraph_numbers: List[np.ndarray],
                table_cell_numbers: List[np.ndarray],
                question_ids: List[str],
                position_ids: torch.LongTensor = None,
                table_mask: torch.LongTensor = None,
                table_cell_index: torch.LongTensor = None,
                table_cell_tokens: List[List[str]] = None,
                mode=None,
                epoch=None, ) -> Dict[str, torch.Tensor]:

        device = input_ids.device

        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids)
        sequence_output = outputs[0]
        batch_size = sequence_output.shape[0]

        cls_output = sequence_output[:, 0, :]

        operator_prediction = self.operator_predictor(cls_output)
        #predicted_operator_class = torch.argmax(operator_prediction, dim=-1)


        opt_output = torch.zeros([batch_size,self.num_ops,self.hidden_size],device = device)

        #opt_output = sequence_output.gather(index = opt_index , dim = 1)
        for bsz in range(batch_size):
            opt_output[bsz] = sequence_output[bsz,opt_mask[bsz]:opt_mask[bsz]+self.num_ops,:]


        ari_ops_prediction = self.ari_predictor(opt_output)
        pred_ari_class = torch.argmax(ari_ops_prediction,dim = -1)

        #print(ari_ops)
        #print(pred_ari_class)
        #print("----------------------------------------------")

        table_sequence_output = util.replace_masked_values(sequence_output, table_mask.unsqueeze(-1), 0)
        table_tag_prediction = self.tag_predictor(table_sequence_output)
        table_tag_prediction = util.masked_log_softmax(table_tag_prediction, mask=None)
        table_tag_prediction = util.replace_masked_values(table_tag_prediction, table_mask.unsqueeze(-1), 0)
        table_tag_labels = util.replace_masked_values(tag_labels.float(), table_mask, 0)

        paragraph_sequence_output = util.replace_masked_values(sequence_output, paragraph_mask.unsqueeze(-1), 0)
        paragraph_tag_prediction = self.tag_predictor(paragraph_sequence_output)
        paragraph_tag_prediction = util.masked_log_softmax(paragraph_tag_prediction, mask=None)
        paragraph_tag_prediction = util.replace_masked_values(paragraph_tag_prediction, paragraph_mask.unsqueeze(-1), 0)
        paragraph_tag_labels = util.replace_masked_values(tag_labels.float(), paragraph_mask, 0)

        paragraph_reduce_mean = torch.mean(paragraph_sequence_output, dim=1)
        table_reduce_mean = torch.mean(table_sequence_output, dim=1)

        scale_output = torch.cat((cls_output, table_reduce_mean, paragraph_reduce_mean), dim=-1)
        scale_prediction = self.scale_predictor(scale_output)
        output_dict = {}

        operator_prediction_loss = self.operator_criterion(operator_prediction, operator_labels)
        scale_prediction_loss = self.scale_criterion(scale_prediction, scale_labels)
        table_tag_prediction = table_tag_prediction.transpose(1, 2)  # [bsz, 2, table_size]
        table_tag_prediction_loss = self.NLLLoss(table_tag_prediction, table_tag_labels.long())
        paragraph_tag_prediction = paragraph_tag_prediction.transpose(1, 2)
        paragraph_token_tag_prediction_loss = self.NLLLoss(paragraph_tag_prediction, paragraph_tag_labels.long())


        output_dict["loss"] = operator_prediction_loss + scale_prediction_loss + table_tag_prediction_loss + paragraph_token_tag_prediction_loss

        ''' 
        pred_ari_class = torch.zeros([batch_size,self.num_ops],device = device)

        for bsz in range(batch_size):
            for roud in range(self.num_ops):
                if ari_ops[bsz,roud] != -100:
                    ari_class_prediction = self.ari_predictor(sequence_output[bsz,opt_mask[bsz]+roud,:])
                    pred_ari_class[bsz,roud] = torch.argmax(ari_class_prediction,dim = -1)
                    output_dict["loss"] = output_dict["loss"] + self.ari_operator_criterion(ari_class_prediction.unsqueeze(0) , ari_ops[bsz,roud].unsqueeze(0))

        pred_ari_class = pred_ari_class.int()
        '''


        if len(torch.nonzero(ari_ops != -100)) != 0:
            output_dict["loss"] =  output_dict["loss"] + self.ari_operator_criterion(ari_ops_prediction.transpose(1,2) , ari_ops)


        if epoch < 4:
            with torch.no_grad():
                predicted_scale_class = torch.argmax(scale_prediction, dim=-1).detach().cpu().numpy()
                output_dict["question_id"] = []
                output_dict["answer"] = [""] * batch_size
                output_dict["scale"] = []
                for bsz in range(batch_size):
                    output_dict["scale"].append(SCALE[int(predicted_scale_class[bsz])])
                    output_dict["question_id"].append(question_ids[bsz])
            return output_dict


        table_tag_prediction = table_tag_prediction.transpose(1, 2)
        paragraph_tag_prediction = paragraph_tag_prediction.transpose(1, 2)
        paragraph_tag_prediction_argmax = torch.argmax(paragraph_tag_prediction, dim=-1).float()
        paragraph_token_tag_prediction = reduce_mean_index(paragraph_tag_prediction_argmax, paragraph_index).detach().cpu().numpy()
        table_tag_prediction_argmax = torch.argmax(table_tag_prediction, dim=-1).float()
        table_cell_tag_prediction = reduce_mean_index(table_tag_prediction_argmax, table_cell_index).detach().cpu().numpy()

        #modify the input_ids
        input_ids_filled = replace_opt(input_ids,opt_mask,self.ari_operator_ids,pred_ari_class,batch_size,self.num_ops)
        #input_ids_filled.requires_grad = True
        outputs_filled = self.encoder(
            input_ids_filled,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids)
        sequence_output_filled = outputs_filled[0]

        opt_output_filled = torch.zeros([batch_size,self.num_ops,self.hidden_size],device = device)

        for bsz in range(batch_size):
            opt_output_filled[bsz] = sequence_output_filled[bsz,opt_mask[bsz]:opt_mask[bsz]+self.num_ops,:]
            para_sel_indexes , paragraph_selected_numbers = get_number_index_from_reduce_sequence(paragraph_token_tag_prediction[bsz],paragraph_numbers[bsz])
            table_sel_indexes , table_selected_numbers = get_number_index_from_reduce_sequence(table_cell_tag_prediction[bsz], table_cell_numbers[bsz])
            selected_numbers = paragraph_selected_numbers + table_selected_numbers
            selected_indexes = para_sel_indexes + table_sel_indexes
            if not selected_numbers:
                continue
            else:
                for selected_index in selected_indexes:
                    for roud in range(self.num_ops):
                        if ari_labels[int(bsz),roud,int(selected_index)] != -100:
                            output_dict["loss"] = output_dict["loss"] + self.ari_criterion(self.operand_predictor(torch.cat((sequence_output_filled[bsz , selected_index , :], opt_output_filled[bsz,roud,:]),dim = -1)),ari_labels[int(bsz),roud,int(selected_index)])




        for i in range(1, self.num_ops):
            for j in range(i):
                if len(torch.nonzero(opt_labels[:,j,i-1] == -100)) < opt_labels.shape[0]:
                    output_dict["loss"] = output_dict["loss"] + self.opt_criterion(
                            self.opt_predictor(torch.cat((opt_output_filled[:, j, :], opt_output_filled[:, i, :]), dim=-1)),opt_labels[:, j, i - 1])

        with torch.no_grad():
            predicted_scale_class = torch.argmax(scale_prediction, dim=-1).detach().cpu().numpy()
            output_dict["question_id"] = []
            output_dict["answer"] = [""]*batch_size
            output_dict["scale"] = []
            for bsz in range(batch_size):
                output_dict["scale"].append(SCALE[int(predicted_scale_class[bsz])])
                output_dict["question_id"].append(question_ids[bsz])

            #print(table_tag_prediction.shape)
            #print(table_tag_prediction)
            #print(table_cell_tag_prediction.shape)
            #print(table_cell_tag_prediction)

            

        return output_dict

    def predict(self,
                input_ids,
                attention_mask,
                token_type_ids,
                paragraph_mask,
                paragraph_index,
                tag_labels,
                gold_answers,
                paragraph_tokens,
                paragraph_numbers,
                table_cell_numbers,
                question_ids,
                opt_mask,
                position_ids=None,
                mode=None,
                epoch=None,
                table_mask=None,
                table_cell_index=None,
                table_cell_tokens=None,
                table_mapping_content=None,
                paragraph_mapping_content=None,
                ):
        device = input_ids.device

        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids)
        sequence_output = outputs[0]
        batch_size = sequence_output.shape[0]

        cls_output = sequence_output[:, 0, :]

        operator_prediction = self.operator_predictor(cls_output)
        predicted_operator_class = torch.argmax(operator_prediction, dim=-1)

        #opt_output = sequence_output.gather(index = opt_index,dim = 1)
        opt_output = torch.zeros([batch_size,self.num_ops,self.hidden_size],device = device)

        for bsz in range(batch_size):
            opt_output[bsz] = sequence_output[bsz,opt_mask[bsz]:opt_mask[bsz]+self.num_ops,:]

        ari_ops_prediction = self.ari_predictor(opt_output)
        pred_ari_class = torch.argmax(ari_ops_prediction,dim = -1)

        table_sequence_output = util.replace_masked_values(sequence_output, table_mask.unsqueeze(-1), 0)
        table_tag_prediction = self.tag_predictor(table_sequence_output)
        table_tag_prediction = util.masked_log_softmax(table_tag_prediction, mask=None)
        table_tag_prediction = util.replace_masked_values(table_tag_prediction, table_mask.unsqueeze(-1), 0)
        table_tag_labels = util.replace_masked_values(tag_labels.float(), table_mask, 0)

        paragraph_sequence_output = util.replace_masked_values(sequence_output, paragraph_mask.unsqueeze(-1), 0)
        paragraph_tag_prediction = self.tag_predictor(paragraph_sequence_output)
        paragraph_tag_prediction = util.masked_log_softmax(paragraph_tag_prediction, mask=None)
        paragraph_tag_prediction = util.replace_masked_values(paragraph_tag_prediction, paragraph_mask.unsqueeze(-1), 0)
        paragraph_tag_labels = util.replace_masked_values(tag_labels.float(), paragraph_mask, 0)

        paragraph_reduce_mean = torch.mean(paragraph_sequence_output, dim=1)
        table_reduce_mean = torch.mean(table_sequence_output, dim=1)

        cls_output = torch.cat((cls_output, table_reduce_mean, paragraph_reduce_mean), dim=-1)
        scale_prediction = self.scale_predictor(cls_output)

        #with torch.no_grad():
        paragraph_tag_prediction_score = paragraph_tag_prediction[:, :, 1]
        paragraph_token_tag_prediction_score = reduce_max_index(paragraph_tag_prediction_score, paragraph_index).detach().cpu().numpy()
        paragraph_tag_prediction_argmax = torch.argmax(paragraph_tag_prediction, dim=-1).float()
        paragraph_token_tag_prediction = reduce_mean_index(paragraph_tag_prediction_argmax, paragraph_index).detach().cpu().numpy()
        table_tag_prediction_score = table_tag_prediction[:, :, 1]
        table_cell_tag_prediction_score = reduce_max_index(table_tag_prediction_score, table_cell_index).detach().cpu().numpy()
        table_tag_prediction_argmax = torch.argmax(table_tag_prediction, dim=-1).float()
        table_cell_tag_prediction = reduce_mean_index(table_tag_prediction_argmax, table_cell_index).detach().cpu().numpy()


        #modify the input_ids
        input_ids = replace_opt(input_ids,opt_mask,self.ari_operator_ids,pred_ari_class,batch_size,self.num_ops)
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids)
        sequence_output = outputs[0]

        opt_output = torch.zeros([batch_size,self.num_ops,self.hidden_size],device = device)
        #opt_output = sequence_output.gather(index = opt_index,dim = 1)
        selected_numbers_output = torch.zeros([200 , self.num_ops, 2*self.hidden_size],device = device)
        number_indexes_batch = np.zeros([200 , 2])

        selected_numbers_batch = []
        #selected_numbers_batch = np.zeros([200])

        num_numbers = 0

        pred_ari_class = pred_ari_class.detach().cpu().numpy()

        predicted_scale_class = torch.argmax(scale_prediction, dim=-1).detach().cpu().numpy()

        output_dict = {}
        output_dict["question_id"] = []
        output_dict["answer"] = []
        output_dict["scale"] = []
        output_dict["gold_answers"] = []
        output_dict["pred_span"] = []
        output_dict["gold_span"] = []

        for bsz in range(batch_size):
            #if gold_answers[bsz]["gold_ops"][0] in  ["Sum","Difference","Multiplication","Division","Average"]:
            #        print(gold_answers[bsz]["gold_ops"])
            #        print(pred_ari_class[bsz])
            #        print("----------------------------------------")
            opt_output[bsz] = sequence_output[bsz,opt_mask[bsz]:opt_mask[bsz]+self.num_ops,:]
            para_sel_indexes , paragraph_selected_numbers = get_number_index_from_reduce_sequence(paragraph_token_tag_prediction[bsz],paragraph_numbers[bsz])
            table_sel_indexes , table_selected_numbers = get_number_index_from_reduce_sequence(table_cell_tag_prediction[bsz], table_cell_numbers[bsz])
            selected_numbers = paragraph_selected_numbers + table_selected_numbers
            selected_indexes = para_sel_indexes + table_sel_indexes

            if not selected_numbers:
                selected_numbers_batch.append([])
                #continue
            else:
                selected_numbers_batch.append(selected_numbers)
                for selected_index in selected_indexes:
                    number_indexes_batch[num_numbers,0] = bsz
                    number_indexes_batch[num_numbers,1] = selected_index
                    #exit(0)
                    for roud in range(self.num_ops):
                        selected_numbers_output[num_numbers,roud] = torch.cat((sequence_output[bsz , selected_index , :], opt_output[bsz,roud,:]),dim = -1)
                    num_numbers += 1

        if num_numbers > 0:
            selected_numbers_output = selected_numbers_output[:num_numbers]
            number_indexes_batch = number_indexes_batch[:num_numbers]
            #selected_numbers_batch = selected_numbers_batch[:num_numbers]
            ari_tags_prediction = self.operand_predictor(selected_numbers_output)
            _ , ari_tags_scores_order = torch.sort(ari_tags_prediction,dim = 0,descending= True)
            opd1_ari_order = ari_tags_scores_order[:,:,1].detach().cpu().numpy()
            opd2_ari_order = ari_tags_scores_order[:,:,2].detach().cpu().numpy()

            pred_ari_tags_class = torch.argmax(ari_tags_prediction,dim = -1).detach().cpu().numpy()
            pred_opt_class = torch.zeros([batch_size,self.num_ops - 1 , self.num_ops - 1],device = device)

            pred_opd1_opt_scores = torch.zeros([batch_size,self.num_ops - 1 , self.num_ops - 1],device = device)
            pred_opd2_opt_scores = torch.zeros([batch_size,self.num_ops - 1 , self.num_ops - 1],device = device)
            for i in range(1,self.num_ops):
                for j in range(i):
                    ari_opt_prediction = self.opt_predictor(torch.cat((opt_output[:,j,:],opt_output[:,i,:]),dim = -1))
                    pred_opd1_opt_scores[:,j,i-1] = ari_opt_prediction[:,1]
                    pred_opd2_opt_scores[:,j,i-1] = ari_opt_prediction[:,2]
                    pred_opt_class[:,j,i-1] = torch.argmax(ari_opt_prediction,dim = -1)
            pred_opt_class = pred_opt_class.detach().cpu().numpy()

            #print(pred_ari_tags_class)
            #print(pred_ari_class)
            #print(pred_opt_class)
            #print("----------------------------------------")
            pred_opd1_opt_scores = pred_opd1_opt_scores.detach().cpu().numpy()
            pred_opd2_opt_scores = pred_opd2_opt_scores.detach().cpu().numpy()

        for bsz in range(batch_size):
            pred_span = []
            current_ops = ["ignore"]* self.num_ops
            if "SPAN-TEXT" in self.OPERATOR_CLASSES and predicted_operator_class[bsz] == self.OPERATOR_CLASSES["SPAN-TEXT"]:
                paragraph_selected_span_tokens = get_single_span_tokens_from_paragraph(
                      paragraph_token_tag_prediction[bsz],
                      paragraph_token_tag_prediction_score[bsz],
                      paragraph_tokens[bsz]
                   )
                answer = paragraph_selected_span_tokens
                answer = sorted(answer)
                output_dict["pred_span"].append(answer)
                pred_span += answer
                current_ops[0] = "Span-in-text"
            elif "SPAN-TABLE" in self.OPERATOR_CLASSES and predicted_operator_class[bsz] == self.OPERATOR_CLASSES["SPAN-TABLE"]:
                table_selected_tokens = get_single_span_tokens_from_table(
                   table_cell_tag_prediction[bsz],
                   table_cell_tag_prediction_score[bsz],
                   table_cell_tokens[bsz])
                answer = table_selected_tokens
                answer = sorted(answer)
                output_dict["pred_span"].append(answer)
                pred_span += answer
                current_ops[0] = "Cell-in-table"
            elif "MULTI_SPAN" in self.OPERATOR_CLASSES and predicted_operator_class[bsz] == self.OPERATOR_CLASSES["MULTI_SPAN"]:
                paragraph_selected_span_tokens = get_span_tokens_from_paragraph(paragraph_token_tag_prediction[bsz], paragraph_tokens[bsz])
                table_selected_tokens = get_span_tokens_from_table(table_cell_tag_prediction[bsz], table_cell_tokens[bsz])
                answer = paragraph_selected_span_tokens + table_selected_tokens
                answer = sorted(answer)
                output_dict["pred_span"].append(answer)
                pred_span += answer
                current_ops[0] = "Spans"
            elif "COUNT" in self.OPERATOR_CLASSES and predicted_operator_class[bsz] == self.OPERATOR_CLASSES["COUNT"]:
                paragraph_selected_tokens = \
                    get_span_tokens_from_paragraph(paragraph_token_tag_prediction[bsz], paragraph_tokens[bsz])
                table_selected_tokens = \
                    get_span_tokens_from_table(table_cell_tag_prediction[bsz], table_cell_tokens[bsz])
                answer = len(paragraph_selected_tokens) + len(table_selected_tokens)
                output_dict["pred_span"].append(answer)
                pred_span += sorted(paragraph_selected_tokens + table_selected_tokens)
                current_ops[0] = "Count"
            else:
                if num_numbers == 0:
                    answer = ""
                else:
                    #selected_numbers = [selected_numbers_batch[i] for i in range(num_numbers) if number_indexes_batch[i,0] == bsz]
                    selected_numbers = selected_numbers_batch[bsz]

                    if len(selected_numbers) == 0:
                        answer = ""
                    else:
                        selected_numbers_labels = [[i,pred_ari_tags_class[i]] for i in range(num_numbers) if number_indexes_batch[i,0] == bsz]
                        #selected_numbers_ids = [i for i in range(num_numbers) if number_indexes_batch[i,0] == bsz]
                        temp_ans = []
                        for roud in range(self.num_ops):
                            roud_selected_numbers = [selected_numbers[i] for i in range(len(selected_numbers)) if selected_numbers_labels[i][1][roud] != 0]
                            if roud > 0 :
                                opt_selected_indexes = pred_opt_class[bsz,:,roud-1]
                                opt_selected_numbers = [temp_ans[i] for i in range(roud) if opt_selected_indexes[i] != 0]
                                roud_selected_numbers += opt_selected_numbers
                            if len(roud_selected_numbers) == 0:
                                print("not roud_selected_numbers")
                                if len(temp_ans) == 0:
                                    answer = ""
                                else:
                                    answer  =temp_ans[-1]
                                #current_ops = ["ignore"] * self.num_ops
                                current_ops[roud] = "Stop"
                                break
                            else:
                                if "STP" in self.ARI_CLASSES and pred_ari_class[bsz,roud] == self.ARI_CLASSES["STP"]:
                                    if roud == 0:
                                        answer = ""
                                        print("stop at first round")
                                        #current_ops = ["ignore"] * self.num_ops
                                        current_ops[roud] = "Stop"
                                    else:
                                        answer = temp_ans[-1]
                                        #current_ops[roud:] = ["Stop"]*(self.num_ops - roud)
                                        current_ops[roud] = "Stop"
                                    break
                                elif "SUM" in self.ARI_CLASSES and pred_ari_class[bsz,roud] == self.ARI_CLASSES["SUM"]:
                                    temp_ans.append(np.around(np.sum(selected_numbers), 4))
                                    current_ops[roud] = "Sum"
                                elif "TIMES" in self.ARI_CLASSES and pred_ari_class[bsz,roud] == self.ARI_CLASSES["TIMES"]:
                                    temp_ans.append(np.around(np.prod(roud_selected_numbers), 4))
                                    current_ops[roud] = "Multiplication"
                                elif "AVERAGE" in self.ARI_CLASSES and pred_ari_class[bsz,roud] == self.ARI_CLASSES["AVERAGE"]:
                                    temp_ans.append(np.around(np.mean(roud_selected_numbers), 4))
                                    current_ops[roud] = "Average"
                                else:
                                    operand_one = np.nan
                                    operand_two = np.nan
                                    if roud > 0 :
                                        opt_selected_indexes = pred_opt_class[bsz,:,roud-1]
                                        opd1_opt_selected_numbers = [[pred_opd1_opt_scores[bsz,i,roud-1],temp_ans[i]] for i in range(roud) if opt_selected_indexes[i] == 1]
                                        if not opd1_opt_selected_numbers:
                                            opd1_roud_selected_numbers = [[selected_numbers_labels[i][0],selected_numbers[i]] for i in range(len(selected_numbers)) if selected_numbers_labels[i][1][roud] == 1]
                                            if not opd1_roud_selected_numbers:
                                                if len(temp_ans) == 0:
                                                    answer = ""
                                                else:
                                                    answer  =temp_ans[-1]
                                                #current_ops = ["ignore"] * self.num_ops
                                                current_ops[roud] = "Stop"
                                                break
                                            else:
                                                min_id = num_numbers
                                                for opd1_number in opd1_roud_selected_numbers:
                                                    sel_number_id_in_order = np.where(opd1_ari_order == opd1_number[0])[0][0]
                                                    if sel_number_id_in_order < min_id:
                                                        operand_one = opd1_number[1]
                                                        min_id = sel_number_id_in_order
                                        else:
                                            best_opt_score = 0
                                            for opd1_opt_number in opd1_opt_selected_numbers:
                                                if opd1_opt_number[0] > best_opt_score:
                                                    operand_one = opd1_opt_number[1]
                                                    best_opt_score = opd1_opt_number[0]

                                        opd2_opt_selected_numbers = [[pred_opd2_opt_scores[bsz,i,roud-1],temp_ans[i]] for i in range(roud) if opt_selected_indexes[i] == 2]
                                        if not opd2_opt_selected_numbers:
                                            opd2_roud_selected_numbers = [[selected_numbers_labels[i][0],selected_numbers[i]] for i in range(len(selected_numbers)) if selected_numbers_labels[i][1][roud] == 2]
                                            if not opd2_roud_selected_numbers:
                                                if len(temp_ans) == 0:
                                                    answer = ""
                                                else:
                                                    answer  =temp_ans[-1]
                                                #current_ops = ["ignore"] * self.num_ops
                                                current_ops[roud] = "Stop"
                                                break
                                            else:
                                                min_id = num_numbers
                                                for opd2_number in opd2_roud_selected_numbers:
                                                    sel_number_id_in_order = np.where(opd2_ari_order == opd2_number[0])[0][0]
                                                    if sel_number_id_in_order < min_id:
                                                        operand_two = opd2_number[1]
                                                        min_id = sel_number_id_in_order
                                        else:
                                            best_opt_score = 0
                                            for opd2_opt_number in opd2_opt_selected_numbers:
                                                if opd2_opt_number[0] > best_opt_score:
                                                    operand_two = opd2_opt_number[1]
                                                    best_opt_score = opd2_opt_number[0]
                                    else:
                                        opd1_roud_selected_numbers = [[selected_numbers_labels[i][0],selected_numbers[i]] for i in range(len(selected_numbers)) if selected_numbers_labels[i][1][roud] == 1]
                                        if not opd1_roud_selected_numbers:
                                            if len(temp_ans) == 0: 
                                               answer = ""
                                            else:
                                               answer  =temp_ans[-1]
                                            #current_ops = ["ignore"] * self.num_ops
                                            current_ops[roud] = "Stop"
                                            break
                                        else:
                                            min_id = num_numbers
                                            for opd1_number in opd1_roud_selected_numbers:
                                                sel_number_id_in_order = np.where(opd1_ari_order == opd1_number[0])[0][0]
                                                if sel_number_id_in_order < min_id:
                                                    operand_one = opd1_number[1]
                                                    min_id = sel_number_id_in_order
                                        opd2_roud_selected_numbers = [[selected_numbers_labels[i][0],selected_numbers[i]] for i in range(len(selected_numbers)) if selected_numbers_labels[i][1][roud] == 2]
                                        if not opd2_roud_selected_numbers:
                                            if len(temp_ans) == 0:
                                                answer = ""
                                            else:
                                                answer  =temp_ans[-1]
                                            #current_ops = ["ignore"] * self.num_ops
                                            current_ops[roud] = "Stop"
                                            break
                                        else:
                                            min_id = num_numbers
                                            for opd2_number in opd2_roud_selected_numbers:
                                                sel_number_id_in_order = np.where(opd2_ari_order == opd2_number[0])[0][0]
                                                if sel_number_id_in_order < min_id:
                                                    operand_two = opd2_number[1]
                                                    min_id = sel_number_id_in_order

                                    if np.isnan(operand_one) or np.isnan(operand_two):
                                        if len(temp_ans) == 0:
                                            answer = ""
                                        else:
                                            answer  =temp_ans[-1]
                                        #current_ops = ["ignore"] * self.num_ops
                                        current_ops[roud] = "Stop"
                                        break
                                    else:
                                        if "DIFF" in self.ARI_CLASSES and pred_ari_class[bsz,roud] == self.ARI_CLASSES["DIFF"]:
                                            temp_ans.append(np.around(operand_one - operand_two, 4))
                                            current_ops[roud] = "Difference"
                                        elif "DIVIDE" in self.ARI_CLASSES and pred_ari_class[bsz,roud] == self.ARI_CLASSES["DIVIDE"]:
                                            temp_ans.append(np.around(operand_one / operand_two, 4))
                                            current_ops[roud] = "Division"
                if answer != "":
                     answer = temp_ans[-1]
                     if SCALE[int(predicted_scale_class[bsz])] == "percent":
                         answer = answer * 100


            output_dict["answer"].append(answer)
            output_dict["scale"].append(SCALE[int(predicted_scale_class[bsz])])
            output_dict["question_id"].append(question_ids[bsz])
            output_dict["gold_answers"].append(gold_answers[bsz])

            #print(current_ops)
            #print(gold_answers[bsz]["gold_ops"])
            #print("---------------------------------------------")
            self._metrics({**gold_answers[bsz], "uid": question_ids[bsz]}, answer,
                          SCALE[int(predicted_scale_class[bsz])], None, None,
                          pred_op=current_ops, gold_op=gold_answers[bsz]["gold_ops"])

        return output_dict

    def reset(self):
        self._metrics.reset()

    def set_metrics_mdoe(self, mode):
        self._metrics = TaTQAEmAndF1(mode=mode)

    def get_metrics(self, logger=None, reset: bool = False) -> Dict[str, float]:
        detail_em, detail_f1 = self._metrics.get_detail_metric()
        raw_detail = self._metrics.get_raw_pivot_table()
        exact_match, f1_score, scale_score, op_score = self._metrics.get_overall_metric(reset)
        print(f"raw matrix:{raw_detail}\r\n")
        print(f"detail em:{detail_em}\r\n")
        print(f"detail f1:{detail_f1}\r\n")
        print(f"global em:{exact_match}\r\n")
        print(f"global f1:{f1_score}\r\n")
        print(f"global scale:{scale_score}\r\n")
        print(f"global op:{op_score}\r\n")
        if logger is not None:
            logger.info(f"raw matrix:{raw_detail}\r\n")
            logger.info(f"detail em:{detail_em}\r\n")
            logger.info(f"detail f1:{detail_f1}\r\n")
            logger.info(f"global em:{exact_match}\r\n")
            logger.info(f"global f1:{f1_score}\r\n")
            logger.info(f"global scale:{scale_score}\r\n")
        return {'em': exact_match, 'f1': f1_score, "scale": scale_score}

    def get_df(self):
        raws = self._metrics.get_raw()
        detail_em, detail_f1 = self._metrics.get_detail_metric()
        raw_detail = self._metrics.get_raw_pivot_table()
        return detail_em, detail_f1, raws, raw_detail


### Beginning of everything related to segmented tensors ###


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


def reduce_mean_vector(values, index, name="segmented_reduce_vector_mean"):
    return _segment_reduce_vector(values, index, "mean", name)


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


def reduce_mean_index_vector(values, index, max_length=512, name="index_reduce_mean"):
    return _index_reduce_vector(values, index, max_length, "mean", name)


def reduce_mean_index(values, index, max_length=512, name="index_reduce_mean"):
    return _index_reduce(values, index, max_length, "mean", name)


def reduce_max_index(values, index, max_length=512, name="index_reduce_max"):
    return _index_reduce_max(values, index, max_length, name)


def reduce_max_index_get_vector(values_for_reduce, values_for_reference, index,
                                max_length=512, name="index_reduce_get_vector"):
    return _index_reduce_max_get_vector(values_for_reduce, values_for_reference, index, max_length, name)


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


def flatten_index(index, max_length=512, name="index_flatten"):
    batch_size = index.shape[0]
    offset = torch.arange(start=0, end=batch_size, device=index.device) * max_length
    offset = offset.view(batch_size, 1)
    return (index + offset).view(-1), batch_size * max_length


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


def _segment_reduce_vector(values, index, segment_reduce_fn, name):
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
    bsz = values.shape[0]
    seq_len = values.shape[1]
    hidden_size = values.shape[2]
    flat_values = values.reshape(bsz * seq_len, hidden_size)
    segment_means = scatter(
        src=flat_values,
        index=flat_index.indices.type(torch.long),
        dim=0,
        dim_size=flat_index.num_segments,
        reduce=segment_reduce_fn,
    )
    output_values = segment_means.view(bsz, -1, hidden_size)
    output_index = range_index_map(index.batch_shape(), index.num_segments)
    return output_values, output_index


def _index_reduce(values, index, max_length, index_reduce_fn, name):
    flat_index, num_index = flatten_index(index, max_length)
    bsz = values.shape[0]
    seq_len = values.shape[1]
    flat_values = values.reshape(bsz * seq_len)
    index_means = scatter(
        src=flat_values,
        index=flat_index.type(torch.long),
        dim=0,
        dim_size=num_index,
        reduce=index_reduce_fn,
    )
    output_values = index_means.view(bsz, -1)
    return output_values


def _index_reduce_max(values, index, max_length, name):
    flat_index, num_index = flatten_index(index, max_length)
    bsz = values.shape[0]
    seq_len = values.shape[1]
    flat_values = values.reshape(bsz * seq_len)
    index_max, _ = scatter_max(
        src=flat_values,
        index=flat_index.type(torch.long),
        dim=0,
        dim_size=num_index,
    )
    output_values = index_max.view(bsz, -1)
    return output_values


def _index_reduce_max_get_vector(values_for_reduce, values_for_reference, index, max_length, name):
    flat_index, num_index = flatten_index(index, max_length)
    bsz = values_for_reduce.shape[0]
    seq_len = values_for_reference.shape[1]
    flat_values_for_reduce = values_for_reduce.reshape(bsz * seq_len)
    flat_values_for_reference = values_for_reference.reshape(bsz * seq_len, -1)
    reduce_values, reduce_index = scatter_max(
        src=flat_values_for_reduce,
        index=flat_index.type(torch.long),
        dim=0,
        dim_size=num_index,
    )
    reduce_index[reduce_index == -1] = flat_values_for_reference.shape[0]
    reduce_values = reduce_values.view(bsz, -1)
    flat_values_for_reference = torch.cat(
        (flat_values_for_reference, torch.zeros(1, flat_values_for_reference.shape[1]).to(values_for_reduce.device)),
        dim=0)
    flat_values_for_reference = torch.index_select(flat_values_for_reference, dim=0, index=reduce_index)
    flat_values_for_reference = flat_values_for_reference.view(bsz, reduce_values.shape[1], -1)
    return reduce_values, flat_values_for_reference


def _index_reduce_vector(values, index, max_length, index_reduce_fn, name):
    flat_index, num_index = flatten_index(index, max_length)
    bsz = values.shape[0]
    seq_len = values.shape[1]
    hidden_size = values.shape[2]
    flat_values = values.reshape(bsz * seq_len, hidden_size)
    index_means = scatter(
        src=flat_values,
        index=flat_index.type(torch.long),
        dim=0,
        dim_size=num_index,
        reduce=index_reduce_fn,
    )
    output_values = index_means.view(bsz, -1, hidden_size)
    return output_values
