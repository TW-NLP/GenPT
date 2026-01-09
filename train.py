"""
模型训练脚本
使用 LoRA 微调 Qwen 模型进行中文 NLU 任务
"""

import os
import gc
import torch
import torch.nn as nn
import warnings
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from datasets import load_dataset, Dataset, load_from_disk
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer, 
    TrainingArguments, 
    DataCollatorWithPadding,
    BitsAndBytesConfig,
    set_seed
)
from peft import (
    get_peft_config, 
    PeftModel, 
    PeftConfig, 
    get_peft_model, 
    LoraConfig, 
    TaskType,
    prepare_model_for_kbit_training
)

warnings.filterwarnings("ignore")


# ==================== 配置参数 ====================
@dataclass
class TrainingConfig:
    """训练配置"""
    # 路径配置
    data_path: str = "./OCEMOTION_train1128"
    model_name: str = "./Qwen3-32B-Instruct"
    output_dir: str = "./outputs_OCEMOTION_All"
    
    # LoRA 配置
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    use_dora: bool = True  # 使用 DoRA (Weight-Decomposed Low-Rank Adaptation)
    lora_target_modules: list = None
    
    # 训练配置
    seed: int = 3407
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4
    warmup_steps: int = 20
    
    # 保存配置
    save_steps: int = 500
    save_total_limit: int = 1
    logging_steps: int = 10
    
    # 量化配置
    load_in_8bit: bool = True
    use_flash_attention: bool = True
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = ['q_proj', 'k_proj', 'v_proj']


# ==================== 工具函数 ====================
def compute_metrics(pred):
    """
    计算评估指标
    
    Args:
        pred: 预测结果，包含 logits 和 labels
    Returns:
        包含 accuracy, precision, recall, f1 的字典
    """
    logits, labels = pred
    preds = logits.argmax(axis=-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro"
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc, 
        'precision': precision, 
        'recall': recall, 
        'f1': f1
    }


def load_model_and_tokenizer(config: TrainingConfig, num_labels: int):
    """
    加载模型和分词器
    
    Args:
        config: 训练配置
        num_labels: 分类标签数量
    Returns:
        model, tokenizer
    """
    print("\n" + "=" * 60)
    print("加载模型和分词器...")
    print("=" * 60)
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # 配置量化
    quantization_config = None
    if config.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
            bnb_8bit_use_double_quant=False,
        )
    
    # 加载模型
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=num_labels,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if config.use_flash_attention else None,
        device_map='auto',
        quantization_config=quantization_config
    )
    
    # 配置模型
    model.config.use_cache = False
    
    print(f"✓ 模型加载完成: {config.model_name}")
    print(f"✓ 标签数量: {num_labels}")
    
    return model, tokenizer


def setup_lora(model, config: TrainingConfig):
    """
    配置 LoRA
    
    Args:
        model: 预训练模型
        config: 训练配置
    Returns:
        配置好 LoRA 的模型
    """
    print("\n" + "=" * 60)
    print("配置 LoRA...")
    print("=" * 60)
    
    # LoRA 配置
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias='none',
        inference_mode=False,
        task_type=TaskType.SEQ_CLS,
        target_modules=config.lora_target_modules,
        use_dora=config.use_dora,  # 启用 DoRA
    )
    
    # 准备模型
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    
    # 打印可训练参数
    model.print_trainable_parameters()
    
    return model


def setup_trainer(model, tokenizer, train_dataset, eval_dataset, config: TrainingConfig):
    """
    配置训练器
    
    Args:
        model: 模型
        tokenizer: 分词器
        train_dataset: 训练数据集
        eval_dataset: 评估数据集
        config: 训练配置
    Returns:
        Trainer 对象
    """
    print("\n" + "=" * 60)
    print("配置训练器...")
    print("=" * 60)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        logging_strategy="steps",
        logging_steps=config.logging_steps,
        warmup_steps=config.warmup_steps,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        fp16=False,
        bf16=True,
        metric_for_best_model="f1",
        report_to="none",
        run_name="qwen_cls",
    )
    
    # 创建 Trainer
    trainer = Trainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    
    print(f"✓ 训练器配置完成")
    print(f"  - 训练轮数: {config.num_train_epochs}")
    print(f"  - 批次大小: {config.per_device_train_batch_size}")
    print(f"  - 梯度累积步数: {config.gradient_accumulation_steps}")
    print(f"  - 学习率: {config.learning_rate}")
    
    return trainer


def main():
    """主函数：执行完整的训练流程"""
    # 初始化配置
    config = TrainingConfig()
    
    # 设置随机种子
    set_seed(config.seed)
    
    print("=" * 60)
    print("开始训练流程")
    print("=" * 60)
    print(f"随机种子: {config.seed}")
    
    # 加载数据集
    print("\n" + "=" * 60)
    print("加载数据集...")
    print("=" * 60)
    
    ds = load_from_disk(config.data_path)
    num_labels = len(set(ds["train"]["labels"]))
    
    print(f"✓ 数据集加载完成")
    print(f"  - 训练样本数: {len(ds['train'])}")
    if "eval" in ds:
        print(f"  - 评估样本数: {len(ds['eval'])}")
    
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(config, num_labels)
    
    # 配置 LoRA
    model = setup_lora(model, config)
    
    # 配置训练器
    trainer = setup_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds.get("eval", None),
        config=config
    )
    
    # 开始训练
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)
    
    trainer.train()
    
    # 保存模型
    print("\n" + "=" * 60)
    print("保存模型...")
    print("=" * 60)
    
    trainer.model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    
    print(f"✓ 模型已保存到: {config.output_dir}")
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
