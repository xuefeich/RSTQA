from transformers import CausalLMOutputWithPast
from transformers.models.llama import LlamaModel,lamaPreTrainedModel
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
from tatqa_metric import TaTQAEmAndF1
from .tools.util import FFNLayer
from .tools import allennlp as util
from typing import Dict, List, Tuple
import numpy as np
from tag_op.data.file_utils import is_scatter_available
from tag_op.data.data_util import SCALE, OPERATOR_CLASSES_,ARITHMETIC_CLASSES_
np.set_printoptions(threshold=np.inf)
# soft dependency
if is_scatter_available():
    from torch_scatter import scatter
    from torch_scatter import scatter_max

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


class LlamaForTAT(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        self.operator_classes = len(OPERATOR_CLASSES_)
        self.ari_classes = len(ARITHMETIC_CLASSES_)
        self.scale_classes = len(SCALE)
        self.num_ops = 6
        hidden_size = config.output_hidden_states
        dropout_prob = 0.1
        self.operator_predictor = FFNLayer(hidden_size, hidden_size, operator_classes, dropout_prob)
        self.ari_predictor = FFNLayer(hidden_size, hidden_size, ari_classes, dropout_prob)
        self.scale_predictor = FFNLayer(3 * hidden_size, hidden_size, scale_classes, dropout_prob)
        self.span_tag_predictor = FFNLayer(hidden_size, hidden_size, 2, dropout_prob)
        self.operand_predictor = FFNLayer(2 * hidden_size, hidden_size, 2, dropout_prob)
        self.opt_predictor = FFNLayer(2 * hidden_size, hidden_size, 3, dropout_prob)
        self.order_predictor = FFNLayer(3 * hidden_size, hidden_size, 2, dropout_prob)
        self.operator_criterion = nn.CrossEntropyLoss()
        self.scale_criterion = nn.CrossEntropyLoss()
        self.ari_criterion = nn.CrossEntropyLoss(reduction = "sum")
        self.opt_criterion = nn.CrossEntropyLoss(reduction = "sum")
        self.order_criterion = nn.CrossEntropyLoss(reduction = "sum")
        self.ari_operator_criterion = nn.CrossEntropyLoss()
        self.NLLLoss = nn.NLLLoss(reduction="sum")
        self.OPERATOR_CLASSES = OPERATOR_CLASSES_
        self.ARI_CLASSES = ARITHMETIC_CLASSES_

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        sequence_output = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(sequence_output, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(sequence_output)
        logits = logits.float()

        loss = None

        batch_size = sequence_output.shape[0]
        cls_output = sequence_output[:, 0, :]
        question_output = util.replace_masked_values(sequence_output, question_mask.unsqueeze(-1), 0)
        question_reduce_mean = torch.mean(question_output, dim=1)
        table_sequence_output = util.replace_masked_values(sequence_output, table_mask.unsqueeze(-1), 0)
        table_tag_prediction = self.span_tag_predictor(table_sequence_output)
        table_tag_prediction = util.masked_log_softmax(table_tag_prediction, mask=None)
        table_tag_prediction = util.replace_masked_values(table_tag_prediction, table_mask.unsqueeze(-1), 0)
        table_tag_labels = util.replace_masked_values(tag_labels.float(), table_mask, 0)

        paragraph_sequence_output = util.replace_masked_values(sequence_output, paragraph_mask.unsqueeze(-1), 0)
        paragraph_tag_prediction = self.span_tag_predictor(paragraph_sequence_output)
        paragraph_tag_prediction = util.masked_log_softmax(paragraph_tag_prediction, mask=None)
        paragraph_tag_prediction = util.replace_masked_values(paragraph_tag_prediction, paragraph_mask.unsqueeze(-1), 0)
        paragraph_tag_labels = util.replace_masked_values(tag_labels.float(), paragraph_mask, 0)

        paragraph_reduce_mean = torch.mean(paragraph_sequence_output, dim=1)
        table_reduce_mean = torch.mean(table_sequence_output, dim=1)

        scale_output = torch.cat((question_reduce_mean,table_reduce_mean, paragraph_reduce_mean), dim=-1)
        operator_prediction = self.operator_predictor(cls_output)
        scale_prediction = self.scale_predictor(scale_output)
        opt_output = torch.zeros([batch_size, self.num_ops, self.hidden_size], device=device)
        for bsz in range(batch_size):
            opt_output[bsz] = sequence_output[bsz,opt_mask[bsz]:opt_mask[bsz]+self.num_ops,:]
        operator_prediction_loss = self.operator_criterion(operator_prediction, operator_labels)
        scale_prediction_loss = self.scale_criterion(scale_prediction, scale_labels)
        table_tag_prediction = table_tag_prediction.transpose(1, 2)  # [bsz, 2, table_size]
        table_tag_prediction_loss = self.NLLLoss(table_tag_prediction, table_tag_labels.long())
        paragraph_tag_prediction = paragraph_tag_prediction.transpose(1, 2)
        paragraph_token_tag_prediction_loss = self.NLLLoss(paragraph_tag_prediction, paragraph_tag_labels.long())
        loss = operator_prediction_loss + scale_prediction_loss + table_tag_prediction_loss + paragraph_token_tag_prediction_loss
        for bsz in range(batch_size):
            for roud in range(self.num_ops):
                if ari_ops[bsz,roud] != -100:
                    loss = loss + self.ari_operator_criterion(self.ari_predictor(opt_output[bsz,roud]).unsqueeze(0) , ari_ops[bsz,roud].unsqueeze(0))

        num_numbers_truth = ari_labels.shape[0]
        selected_numbers_output = torch.zeros([num_numbers_truth,self.num_ops,2*self.hidden_size],device = device)
        num_numbers = 0
        order_numbers = []
        if num_numbers_truth >0:
            for bsz in range(batch_size):
               order_numbers.append([])
               for selected_index in selected_indexes:
                   if selected_index[0] == bsz:
                       k = np.where(selected_index[1:] == 0)[0] # [bsz,subtok_index , ....,0]
                       if len(k) == 0:
                           number_index = selected_index[1:]
                       else:
                           number_index = selected_index[1:k[0]+1]
                       for roud in range(self.num_ops):
                           order_numbers[bsz].append([])
                           selected_numbers_output[num_numbers,roud] = torch.cat((torch.mean(sequence_output[bsz , number_index],dim = 0), opt_output[bsz,roud]),dim = -1)
                           if ari_labels[num_numbers,roud] == 1:
                               order_numbers[bsz][roud].append(number_index)
                       num_numbers += 1

            operand_prediction = self.operand_predictor(selected_numbers_output)
            operand_loss = self.ari_criterion(operand_prediction.transpose(1,2),ari_labels)
            loss = loss + operand_loss

        if len(torch.nonzero(order_labels == -100)) < batch_size * self.num_ops:
            order_output = torch.zeros([batch_size,self.num_ops,3*self.hidden_size],device = device)
            for bsz in range(batch_size):
               for roud in range(self.num_ops):
                  if order_labels[bsz,roud] != -100:
                     opd1_output = torch.mean(sequence_output[bsz , order_numbers[bsz][roud][0]],dim = 0)
                     opd2_output = torch.mean(sequence_output[bsz , order_numbers[bsz][roud][1]],dim = 0)
                     order_output[bsz,roud] = torch.cat((opd1_output, opt_output[bsz,roud] , opd2_output),dim = -1)

            order_prediction = self.order_predictor(order_output)
            order_loss = self.order_criterion(order_prediction.transpose(1,2),order_labels)
            loss = loss + order_loss

        for i in range(1, self.num_ops):
            for j in range(i):
                if len(torch.nonzero(opt_labels[:,j,i-1] == -100)) < opt_labels.shape[0]:
                    loss = loss + self.opt_criterion(
                            self.opt_predictor(torch.cat((opt_output[:, j, :], opt_output[:, i, :]), dim=-1)),opt_labels[:, j, i - 1])

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
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








