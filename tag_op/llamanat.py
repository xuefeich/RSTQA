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
