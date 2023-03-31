"""
    基于layoutlmv3 和 global pointer 进行实体识别+关系抽取;
    # 此版本依托icdar2023 t1 任务的数据来进行示例。
    # 支持 train / eval
    # 不支持单独的 predicrt, 另外有单独的 predict 脚本
                    by zhangzhi600@20230329

debug运行参数:
--data_dir /home/ana/data2/datasets/icdar2023.challenge/icdar2023/task1/train_test_spl_final
--do_train --do_eval
--model_name_or_path /home/ana/data2/models/layoutlmv3-base-chinese --output_dir t1.rel.tidy.debug
--segment_level_layout 1 --visual_embed 1 --input_size 224 --max_steps 10 --save_steps 5
--evaluation_strategy steps --eval_steps 2 --logging_steps 2 --learning_rate 7e-5
--per_device_train_batch_size 2 --per_device_eval_batch_size 2 --eval_accumulation_steps 2
--gradient_accumulation_steps 2 --overwrite_output_dir
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

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
from libs.GlobalPointer import MetricsCalculator, GlobalPointerRelModel

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# todo: 目前的实现版本其实也不能太高，会和官方的layoutlmv3 有冲突，后续看看如何解决
check_min_version("4.5.0")
# root logger
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
    # 可以用于继续训练或者预测(默认最后一个ckpt)
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
    # xlm roberta
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
        # train_dataset = t1_dataset_gp(data_args, tokenizer, 'train', image_base_path, real_ocr_path=real_ocr_path)
        train_dataset = t1_dataset_gp(data_args, tokenizer, 'train', image_base_path, real_ocr_path=real_ocr_path,
                                      max_records=60)
        logger.info(f"train_dataset loaded: len: {len(train_dataset)}")
    #
    if training_args.do_eval:
        # eval_dataset = t1_dataset_gp(data_args, tokenizer, 'eval', image_base_path, real_ocr_path=real_ocr_path)
        eval_dataset = t1_dataset_gp(data_args, tokenizer, 'eval', image_base_path, real_ocr_path=real_ocr_path,
                                     max_records=60)
        logger.info(f"eval_dataset loaded: len: {len(eval_dataset)}")

    if training_args.do_predict:
        raise NotImplementedError("不支持predict")
        # 改一下
        test_ids_order = [l.strip() for l in open(os.path.join(data_args.data_dir, "test_ids.txt")).read().split(
            "\n") if l.strip() != ""]
        # test_ids_order = test_ids_order[-4:]
        real_ocr_path = os.path.join(data_args.data_dir, "task1_recg_text")
        # test_dataset 的 real_ocr 需要时icdar2023的官方格式
        test_dataset = t1_test_dataset_gp(data_args, test_ids_order, tokenizer, image_base_path,
                                          real_ocr_path = real_ocr_path)
        #                                  real_ocr_path=real_ocr_path, max_records=10)
        logger.info(f"test_dataset loaded: len: {len(test_dataset)}")

    # v3
    encoder = LayoutLMv3Model.from_pretrained(
        model_args.model_name_or_path,
        # from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # global linker model
    model = GlobalPointerRelModel(encoder, config, ent_type_size=len(label2ids_span_gp),
                                  rel_type_size=len(rel_types_sp), fixed_text_len=512, RoPE=True,
                                  RoPE_dim=2)
    # 手工
    logger.info(f"last_checkpoint is {last_checkpoint}")
    if last_checkpoint is not None:
        logger.info(f"loading MODEL weight from {last_checkpoint}")
        state_dict = torch.load(os.path.join(last_checkpoint, 'pytorch_model.bin'))
        model.load_state_dict(state_dict, strict=False)

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )

    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length"

    # Data collator for global pointer
    # 核心的label 处理工作在这里
    data_collator = DataCollatorForKeyValueExtractionGp(
        tokenizer,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        padding=padding,
        max_length=512,
        num_labels=len(label2ids_span_gp),
        num_rel_labels=len(rel_types_sp),
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
        # todo: 端到端结果后续再补充
        all_f1, all_precision, all_recall = metric_helper.get_evaluate_fpr(predictions[1], labels)
        ent_f1, ent_precision, ent_recall = metric_helper.get_evaluate_fpr(
            predictions[1][:, :model.ent_type_size, :, :], labels[:, :model.ent_type_size, :, :])
        head_f1, head_precision, head_recall = metric_helper.get_evaluate_fpr(
            predictions[1][:, model.ent_type_size:model.ent_type_size + model.rel_type_size, :, :],
            labels[:, model.ent_type_size:model.ent_type_size + model.rel_type_size, :, :])
        tail_f1, tail_precision, tail_recall = metric_helper.get_evaluate_fpr(
            predictions[1][:, -model.rel_type_size:, :, :], labels[:, -model.rel_type_size:, :, :])
        # model = GlobalPointer(encoder, config, len(train_dataset.label2ids), fixed_text_len=512)
        return {
            "precision": all_precision,
            "recall": all_recall,
            "f1": all_f1,
            "ent_f1": ent_f1,
            "ent_precision": ent_precision,
            "ent_recall": ent_recall,
            "head_f1": head_f1,
            "head_precision": head_precision,
            "head_recall": head_recall,
            "tail_f1": tail_f1,
            "tail_precision": tail_precision,
            "tail_recall": tail_recall
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
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
