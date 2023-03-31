#!/usr/bin/env python
# coding=utf-8
"""
    使用 global pointer 来完成 t1 中的 ner 任务

    参考debug 参数
    --data_dir /home/ana/data2/datasets/icdar2023.challenge/icdar2023/task1/train_test_spl
    --do_train --do_eval --model_name_or_path /home/ana/data2/models/layoutlmv3-base-chinese
    --output_dir t1.output.gp.ddp --segment_level_layout 1 --visual_embed 1 --input_size 224
    --max_steps 10 --save_steps 5 --evaluation_strategy steps --eval_steps 5 --logging_steps 5
    --learning_rate 7e-5 --per_device_train_batch_size 2 --per_device_eval_batch_size 2
    --eval_accumulation_steps 2 --gradient_accumulation_steps 4 --overwrite_output_dir


"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# from datasets import ClassLabel, load_dataset, load_metric
import torch
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

import kp_setup
from libs.layoutlmv3.layoutlmft.data import DataCollatorForKeyValueExtractionGp
from libs.datasets.t1_dataset_gp import t1_dataset_gp, t1_test_dataset_gp, label2ids_span_gp, rel_types_sp
from libs.layoutlmv3.layoutlmft.models import LayoutLMv3Model
from libs.GlobalPointer import MetricsCalculator, GlobalPointerNerModel


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.5.0")
logger = logging.getLogger()


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(default="ner", metadata={"help": "The name of the task (ner, pos...)."})
    language: Optional[str] = field(
        default='zh', metadata={"help": "The dataset in xfund to use"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                    "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
                    "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
                    "value if set."
        },
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether to put the label for one word on all tokens of generated by that word or just on the "
                    "one (in which case the other tokens will have a padding index)."
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )
    segment_level_layout: bool = field(default=True)
    visual_embed: bool = field(default=True)
    data_dir: Optional[str] = field(default=None)
    input_size: int = field(default=224, metadata={"help": "images input size for backbone"})
    second_input_size: int = field(default=112, metadata={"help": "images input size for discrete vae"})
    train_interpolation: str = field(
        default='bicubic', metadata={"help": "Training interpolation (random, bilinear, bicubic)"})
    second_interpolation: str = field(
        default='lanczos', metadata={"help": "Interpolation for discrete vae (random, bilinear, bicubic)"})
    imagenet_default_mean_and_std: bool = field(default=False, metadata={"help": ""})


def main():
    # See all possible arguments in layoutlmft/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    if training_args.do_predict:
        # todo 给test 的ckpt一个独立的参数
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=3,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        input_size=data_args.input_size,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        tokenizer_file=None,  # avoid loading from a cached file of the pre-trained model in another machine
        cache_dir=model_args.cache_dir,
        use_fast=True,
        add_prefix_space=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    train_dataset, eval_dataset, test_dataset = None, None, None
    image_base_path = os.path.join(data_args.data_dir, 'images')
    real_ocr_path = os.path.join(data_args.data_dir, 'pp_images')
    if training_args.do_train:
        train_dataset = t1_dataset_gp(data_args, tokenizer, 'train', image_base_path, real_ocr_path=real_ocr_path, max_records=50)
        # train_dataset = t1_dataset_gp(data_args, tokenizer, 'train', image_base_path, real_ocr_path=None)
        logger.info(f"train_dataset loaded: len: {len(train_dataset)}")
    if training_args.do_eval:
        eval_dataset = t1_dataset_gp(data_args, tokenizer, 'eval', image_base_path, real_ocr_path=real_ocr_path, max_records=20)
        # eval_dataset = t1_dataset_gp(data_args, tokenizer, 'eval', image_base_path, real_ocr_path=None)
        logger.info(f"eval_dataset loaded: len: {len(eval_dataset)}")
    if training_args.do_predict:
        logger.warning("用eval数据集做个样例，实践上可以参考 run_er_extraction_predict")
        test_dataset = t1_dataset_gp(data_args, tokenizer, 'eval', image_base_path, real_ocr_path=real_ocr_path, max_records=20)
        logger.info(f"eval_dataset loaded: len: {len(eval_dataset)}")

    encoder = LayoutLMv3Model.from_pretrained(
        model_args.model_name_or_path,
        # from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = GlobalPointerNerModel(encoder, config, len(label2ids_span_gp), fixed_text_len=512, RoPE=True, RoPE_dim=2)
    # 手工
    logger.info(f"last_checkpoint is {last_checkpoint}")
    if last_checkpoint is not None:
        logger.info(f"loading MODEL weight from {last_checkpoint}")
        state_dict = torch.load(os.path.join(last_checkpoint, 'pytorch_model.bin'))
        model.load_state_dict(state_dict, strict=False)

    # layer_names = []
    # for idx, (name, param) in enumerate(model.named_parameters()):
    #     layer_names.append(name)
    #     print(f'{idx}: {name}')

    # sys.exit(0)

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )

    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False
    # Data collator for global pointer
    # 核心的label 处理工作在这里
    data_collator = DataCollatorForKeyValueExtractionGp(
        tokenizer,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        padding=padding,
        max_length=512,
        num_labels=len(label2ids_span_gp),
        is_ent_task_only=True,
        fixed_text_length=512
    )

    def get_label_list():
        label_list = [[key, val] for key, val in label2ids_span_gp.items()]
        label_list = sorted(label_list, key=lambda x: x[1], reverse=False)
        label_list = [label for label, id in label_list]
        return label_list

    label_list = get_label_list()

    # Metrics
    # metric = load_metric("seqeval")
    metric_helper = MetricsCalculator()

    def compute_metrics(p):
        # 这里面pred 默认是啥？
        predictions, labels = p
        # 要跟进来看看
        all_f1, all_precision, all_recall = metric_helper.get_evaluate_fpr(predictions[1], labels)
        return {
            "precision": all_precision,
            "recall": all_recall,
            "f1": all_f1,
        }

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        logger.info("training BEGIN")
        checkpoint = last_checkpoint if last_checkpoint else None
        # train_result = trainer.train(resume_from_checkpoint=checkpoint)
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        logger.info("training END")

    # Evaluation
    if training_args.do_eval and False:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predictions, labels, metrics = trainer.predict(test_dataset)

        #print(predictions)
        #predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

        pred = []
        for b, l, start, end in zip(*np.where(predictions[1] > 0)):
            result = None
            if labels is not None:
                if labels[b,l,start,end] == 1:
                    result = 1
                else:
                    result = 0
            if end >= start:
                pred.append((b, label_list[l], start, end, result))

        # Save predictions
        output_test_predictions_file = os.path.join(training_args.output_dir, "test_predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_test_predictions_file, "w") as writer:
                for prediction in pred:
                    writer.write(" ".join([str(item) for item in prediction]) + "\n")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
