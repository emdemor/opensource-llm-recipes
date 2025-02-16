import os

if os.environ.get("TRANSFORMERS_CACHE"):
    os.environ["HF_HOME"] = os.environ.pop("TRANSFORMERS_CACHE")

import math
import re
import shutil
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, List, Optional

import flash_attn
import pandas as pd
import tabulate
import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from flash_attn import flash_attn_qkvpacked_func
from huggingface_hub import Repository, whoami
from pydantic import BaseModel, field_validator
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)


AMOUNT_OF_NEWS = 3535
AMOUNT_OF_SENTENCES = 3456

MODEL_ID = "answerdotai/ModernBERT-base"  # "neuralmind/bert-base-portuguese-cased"
DATASET_ID = "emdemor/news-of-the-brazilian-newspaper"
USERNAME = "emdemor"
TOKENIZER_PATH = "domain_tokenizer"
TESTING = True
FLASH_ATTENTION = False
PUSH_INTERVAL = 10_000 if TESTING else 100_000
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

MODEL_NAME = MODEL_ID.split("/")[-1]
TRAINED_MODEL_PATH = f"{MODEL_NAME}-ptbr-{'test' if TESTING else 'full'}"

MLM_PROBABILITIES = [0.05, 0.10, 0.15, 0.20, 0.30]


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]


def set_attention(model):
    def check_flash_attention_support():
        if not torch.cuda.is_available():
            return False
        try:
            qkv = torch.randn(1, 1, 3, 16, 64, dtype=torch.float16, device="cuda")
            flash_attn_qkvpacked_func(qkv, causal=False)
            return True
        except RuntimeError as e:
            print("Flash Attention não é compatível:", str(e))
            return False

    if FLASH_ATTENTION and check_flash_attention_support():
        print("Replacing standard attention with FlashAttention...")
        for module in model.modules():
            if isinstance(module, nn.MultiheadAttention):
                module.attention = FlashAttention()
        print("FlashAttention integrated.")

    return model


def check_vocab_size(tokenizer, model):
    max_token_id = max(tokenizer.get_vocab().values())
    print("Maior ID no tokenizador:", max_token_id)
    print("Tamanho do vocabulário do modelo:", model.config.vocab_size)
    assert max_token_id < model.config.vocab_size, "IDs de tokens fora do intervalo!"


def tokenize_function(examples, target_column="text"):
    return tokenizer(
        examples[target_column],
        # No truncation and max_length to allow dynamic padding truncation=True, max_length=chunk_size, padding="longest",
        return_special_tokens_mask=True,
    )


def tokenize_dataset(dataset):

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    return tokenized_dataset


class TrainingConfig(BaseModel):
    dataset_size: int
    num_train_epochs: int
    num_chunks: int
    train_batch_size_per_device: int
    gradient_accumulation_steps: int
    eval_size_ratio: float
    total_save_limit: int

    @field_validator("num_chunks")
    def validate_num_chunks(cls, v, info):
        data = info.data
        if (
            "dataset_size" in data
            and "dataset_size" in data
            and "eval_size_ratio" in data
        ):
            dataset_size = data["dataset_size"]
            eval_size_per_chunk = int(data["dataset_size"] * data["eval_size_ratio"])
            available_size = dataset_size - eval_size_per_chunk * v
            if available_size < v:
                raise ValueError(
                    f"available_size ({available_size}) deve ser maior ou igual a num_chunks ({v})"
                )
        return v

    @property
    def effective_batch_size(self):
        return self.train_batch_size_per_device * self.gradient_accumulation_steps

    @property
    def total_steps_per_epoch(self):
        return math.ceil(self.dataset_size / self.effective_batch_size)

    @property
    def total_train_steps(self):
        return self.total_steps_per_epoch * self.num_train_epochs

    @property
    def eval_size_per_chunk(self):
        """Tamanho do dataset de avaliação em cada chunk"""
        return int(self.dataset_size * self.eval_size_ratio / self.num_chunks)

    @property
    def available_size(self):
        return self.dataset_size - self.eval_size_per_chunk * self.num_chunks

    @property
    def eval_size(self):
        return self.dataset_size - self.available_size

    @property
    def chunk_size(self):
        return self.dataset_size // self.num_chunks

    @property
    def chunk_train_size(self):
        return self.available_size // self.num_chunks

    def __repr(self):
        data = [
            ["num_train_epochs", self.num_train_epochs],
            ["dataset_size", self.dataset_size],
            ["num_chunks", self.num_chunks],
            ["chunk_size", self.chunk_size],
            ["chunk_train_size", self.chunk_train_size],
            ["eval_size_per_chunk", self.eval_size_per_chunk],
            ["eval_size_ratio", self.eval_size_ratio],
            ["available_size", self.available_size],
            ["eval_size", self.eval_size],
            ["train_batch_size_per_device", self.train_batch_size_per_device],
            ["gradient_accumulation_steps", self.gradient_accumulation_steps],
            ["total_save_limit", self.total_save_limit],
            ["effective_batch_size", self.effective_batch_size],
            ["total_steps_per_epoch", self.total_steps_per_epoch],
            ["total_train_steps", self.total_train_steps],
        ]

        return tabulate.tabulate(data, headers=["Attribute", "Value"], tablefmt="grid")

    def __repr__(self):
        return self.__repr()

    def __str__(self):
        return self.__repr()


# --- Helper Function to Fix Batch Inputs ---
def fix_batch_inputs(inputs: dict) -> dict:
    """
    Esta função tem como objetivo garantir que os tensores de entrada tenham a forma e o tipo corretos:

    - Ela verifica três chaves importantes: "input_ids", "attention_mask" e "token_type_ids"
    - Remove dimensões extras (por exemplo, converte [1, batch, seq_len] para [batch, seq_len])
    - Converte "input_ids" para o tipo torch.long, que é o tipo esperado para IDs de tokens
    - Isso é importante porque inconsistências na forma dos tensores podem causar erros durante o treinamento
    """
    for key in ["input_ids", "attention_mask", "token_type_ids"]:
        if key in inputs:
            if inputs[key].dim() == 3 and inputs[key].shape[0] == 1:
                inputs[key] = inputs[key].squeeze(0)
            elif inputs[key].dim() > 2:
                raise ValueError(
                    f"Unexpected tensor shape for {key}: {inputs[key].shape}"
                )
    if "input_ids" in inputs and inputs["input_ids"].dtype != torch.long:
        inputs["input_ids"] = inputs["input_ids"].long()
    return inputs


# --- Forward Pass Function ---
def forward_pass(model, inputs):
    """
    Esta função realiza uma passagem para frente (forward pass) no modelo:

    - Primeiro, aplica a função fix_batch_inputs para garantir que as entradas estão corretas
    - Move os tensores para o dispositivo apropriado (CPU ou GPU)
    - Usa torch.amp.autocast para habilitar precisão mista automática quando estiver usando GPU
      - A precisão mista acelera o treinamento e reduz o uso de memória
    - Executa o modelo com as entradas e solicita que retorne um dicionário completo
    - Verifica se o modelo retornou uma perda (loss) e a retorna
    """
    inputs = fix_batch_inputs(inputs)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
        outputs = model(**inputs, return_dict=True)
    if outputs.loss is None:
        raise ValueError("Model did not return a loss.")
    return outputs.loss


# --- Evaluation Function ---
def evaluate(model, eval_dataset, data_collator, batch_size):
    """
    Esta função avalia o desempenho do modelo no conjunto de dados de avaliação:

    - Coloca o modelo em modo de avaliação (model.eval())
    - Itera sobre o conjunto de dados de avaliação em lotes
    - Para cada lote:
      - Desativa o cálculo de gradientes com torch.no_grad()
      - Usa precisão mista se estiver em GPU
      - Calcula a perda e a adiciona à lista de perdas
      - Captura e imprime erros, mas continua a avaliação
    - Retorna ao modo de treinamento (model.train())
    - Calcula e retorna a perda média
    """
    model.eval()
    losses = []
    eval_iterator = eval_dataset.iter(batch_size=batch_size)
    for batch in tqdm(eval_iterator, desc="Evaluating"):
        with torch.no_grad(), torch.amp.autocast(
            "cuda", enabled=(DEVICE.type == "cuda")
        ):
            inputs = data_collator(batch)
            try:
                loss = forward_pass(model, inputs)
                losses.append(loss.item())
            except Exception as e:
                print(f"Evaluation batch failed: {e}. Skipping.")
                continue
    model.train()
    average_loss = sum(losses) / len(losses) if losses else float("inf")
    return average_loss


class DynamicPaddingDataCollator(DataCollatorForLanguageModeling):
    """
    Esta classe estende DataCollatorForLanguageModeling e implementa um colator de dados com preenchimento dinâmico:

    - O preenchimento dinâmico significa que cada lote é preenchido apenas até o comprimento da sequência mais longa naquele lote específico
    - Isso é mais eficiente que usar um comprimento fixo para todos os lotes
    - Para cada exemplo no lote:
      - Calcula quanto padding é necessário
      - Adiciona tokens de padding aos IDs de entrada e zeros às máscaras de atenção
      - Ou trunca se necessário
    - Aplica a lógica de colação de dados do MLM (mascaramento aleatório de tokens)
    - Garante formas e tipos corretos com fix_batch_inputs
    """

    def __call__(self, examples: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        # Find the maximum length within the current batch
        max_length = max(len(input_ids) for input_ids in examples["input_ids"])

        # Pad or truncate each example to the max_length
        batch = []
        input_ids = examples["input_ids"]
        attention_mask = examples["attention_mask"]

        for ids, mask in zip(input_ids, attention_mask):
            padding_length = max_length - len(ids)
            if padding_length > 0:
                # Pad
                ids = torch.tensor(ids + [self.tokenizer.pad_token_id] * padding_length)
                mask = torch.tensor(mask + [0] * padding_length)
            elif padding_length <= 0:
                # Truncate (if enabled in your tokenizer)
                ids = torch.tensor(ids[:max_length])
                mask = torch.tensor(mask[:max_length])

            batch.append({"input_ids": ids, "attention_mask": mask})

        # Apply the rest of the data collation logic (MLM masking, etc.)
        batch = self.torch_call(
            batch
        )  # Use torch_call instead of __call__ to call the parent's method

        # Ensure correct shapes and dtypes
        batch = fix_batch_inputs(batch)

        return batch


def start_log(epoch, chunk_number, training_config):
    print(
        f"\nEpoch {epoch + 1}/{training_config.num_train_epochs} | "
        f"MLM Probability: {MLM_PROBABILITIES[chunk_number]}"
    )


def chunk_train_test_split(chunk_number, training_config):

    eval_start_idx = chunk_number * training_config.chunk_size
    eval_end_idx = eval_start_idx + training_config.eval_size_per_chunk - 1
    train_start_idx = (
        chunk_number * training_config.chunk_size + training_config.eval_size_per_chunk
    )
    train_end_idx = train_start_idx + training_config.chunk_train_size - 1

    print(
        f"\tSplitting | "
        f"chunk: {eval_start_idx}-{train_end_idx} | "
        f"eval: {eval_start_idx}-{eval_end_idx} | "
        f"train: {train_start_idx}-{train_end_idx}"
    )

    train_dataset = (
        tokenized_dataset.skip(train_start_idx)
        .take(training_config.chunk_train_size)
        .shuffle(seed=42)
    )

    eval_dataset = (
        tokenized_dataset.skip(eval_start_idx)
        .take(training_config.eval_size_per_chunk)
        .shuffle(seed=42)
    )

    return train_dataset, eval_dataset


def eval_loss(model, batch):
    inputs = data_collator(batch)
    loss = forward_pass(model, inputs)
    return loss


def update_model_parameters(scaler, optimizer, scheduler):
    """Atualiza os parâmetros do modelo, ajusta o learning rate e limpa os gradientes."""
    global global_step
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    optimizer.zero_grad()
    torch.cuda.empty_cache()  # Limpa a memória da GPU
    global_step += 1


def evaluate_step(model, eval_dataset, data_collator, training_config):
    global global_step
    eval_interval = training_config.total_steps_per_epoch // (
        training_config.num_train_epochs * 4
    )
    if eval_interval > 0 and (global_step % eval_interval == 0):
        eval_loss = evaluate(
            model,
            eval_dataset,
            data_collator,
            batch_size=training_config.train_batch_size_per_device,
        )
        print(f"Evaluation loss at step {global_step}: {eval_loss}")


def push_to_hub(tokenizer, model):
    global global_step
    # Push to hub incl TESTING
    if global_step % PUSH_INTERVAL == 0:
        print(f"Saving and pushing model at step {global_step}...")
        model.save_pretrained(TRAINED_MODEL_PATH)
        tokenizer.save_pretrained(TRAINED_MODEL_PATH)
        print(f"Model saved and pushed at step {global_step}.")


### Preparar a Base de Dados


"""Load and prepare the dataset for training."""

raw_dataset = load_dataset(DATASET_ID, split="train")
df = raw_dataset.to_pandas().sample(frac=1).reset_index(drop=True)
sample_df = df.sample(min(AMOUNT_OF_NEWS, len(df)))
combined_texts = sample_df["text"].to_list() + sample_df["title"].to_list()
sentences = [
    phrase for text in combined_texts if text for phrase in split_into_sentences(text)
]
sentences_sample = pd.Series(sentences).sample(AMOUNT_OF_SENTENCES).to_list()
dataset = Dataset.from_dict({"text": sentences_sample})


### Setup model and tokenizer


tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_PATH, clean_up_tokenization_spaces=False
)
config = AutoConfig.from_pretrained(MODEL_ID)
config.torch_dtype = torch.float16
model = AutoModelForMaskedLM.from_pretrained(MODEL_ID, config=config)
model.resize_token_embeddings(len(tokenizer))
model.to(DEVICE)
model = set_attention(model)
check_vocab_size(tokenizer, model)


training_config = TrainingConfig(
    num_train_epochs=3,
    dataset_size=len(dataset),
    num_chunks=len(MLM_PROBABILITIES),
    train_batch_size_per_device=4,
    gradient_accumulation_steps=2,
    eval_size_ratio=0.10,
    total_save_limit=2,
    estimated_dataset_size_in_rows=len(dataset),
)


print(training_config)


tokenized_dataset = tokenize_dataset(dataset)

print(f"Tamanho do dataset = {training_config.dataset_size}")
print(
    f"Serão definidos {training_config.num_chunks} chunks de {training_config.chunk_size} dados"
)
print(
    f"Cada chunk terá {training_config.chunk_train_size} dados de treino e {training_config.eval_size_per_chunk} dados de validação."
)
print(f"O número de dados de treino será, portanto: {training_config.available_size}")
print(f"O número de dados de validação será: {training_config.eval_size}")


# Treinamento


LEARNING_RATE = 5e-3
WEIGHT_DECAY = 0.01
NUM_WARMUP_STEPS = 0


optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

scaler = torch.amp.GradScaler(DEVICE, enabled=(DEVICE.type == "cuda"))

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=NUM_WARMUP_STEPS,
    num_training_steps=training_config.total_train_steps,
)


model.train()
global_step = 0

for epoch in range(training_config.num_train_epochs):
    for chunk_number, mlm_probability in enumerate(MLM_PROBABILITIES):
        start_log(epoch, chunk_number, training_config)
        data_collator = DynamicPaddingDataCollator(
            tokenizer, mlm_probability=mlm_probability
        )
        train_dataset, eval_dataset = chunk_train_test_split(
            chunk_number, training_config
        )
        train_iterator = train_dataset.iter(
            batch_size=training_config.train_batch_size_per_device
        )
        for step, batch in tqdm(
            enumerate(train_iterator), desc=f"Training (MLM {mlm_probability})"
        ):
            accumulation_step_complete = (
                step + 1
            ) % training_config.gradient_accumulation_steps == 0
            try:
                loss = eval_loss(model, batch)
                scaler.scale(
                    loss / training_config.gradient_accumulation_steps
                ).backward()
                if accumulation_step_complete:
                    update_model_parameters(scaler, optimizer, scheduler)
                    evaluate_step(model, eval_dataset, data_collator, training_config)
                    push_to_hub(tokenizer, model)

            except Exception as e:
                print(f"Training batch failed: {e}. Skipping.")
                continue
