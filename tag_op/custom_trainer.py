from transformers import trainer
from transformers.integrations.deepspeed import deepspeed_init
from transformers.trainer_utils import EvalLoopOutput, has_length
from transformers.trainer_pt_utils import find_batch_size
from .utils import is_torch_npu_available
from RSTQA.tatqa_metric import extract_gold_answers, get_answer_str,add_percent_pred,metric_max_over_ground_truths,get_metrics
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union


class F1LoopOutput(NamedTuple):
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]

class TATTrainer(Trainer):
   def compute_f1(self, model, inputs):
        labels = inputs.pop("labels")
        outputs = model.predict(**inputs)
        batch_size = len(labels)
        f1 = 0
        for bsz in range(batch_size):
            ground_truth = labels[bsz]["ground_truth"]
            prediction = outputs[bsz]["prediction"]
            pred_scale = outputs[bsz]["pred_scale"]
            gold_type, gold_answer, gold_scale = extract_gold_answers(ground_truth)
            if gold_answer:
                ground_truth_answer_strings = get_answer_str(gold_answer, gold_scale)
                prediction = prediction if isinstance(prediction, list) else [prediction]
                prediction_strings = get_answer_str(prediction, pred_scale)
                prediction_strings = add_percent_pred(prediction_strings, pred_scale, prediction)
                exact_match, fscore = metric_max_over_ground_truths(
                   get_metrics,
                   prediction_strings,
                   ground_truth_answer_strings
                )
                f1 += fscore
      
        return f1

   def eval_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
    ):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                f1 = self.compute_f1(model, inputs)
        return loss

   def eval_f1_loop(
        self,
        dataloader: DataLoader,
        description: str,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        args = self.args

        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()
        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop

        all_f1 = 0.0
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            all_f1 += self.eval_step(model, inputs)
            if is_torch_tpu_available():
                xm.mark_step()

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            delattr(self, "_past")

        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        metrics = {}
        metrics["f1"] = f1/num_samples
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return F1LoopOutput(metrics=metrics, num_samples=num_samples)


