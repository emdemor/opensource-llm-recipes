import math

from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from .config import MLMTrainingConfig


class TrainingParams:
    """Calcula parâmetros derivados para o treinamento"""

    def __init__(self, config: MLMTrainingConfig):
        self.num_train_epochs = config.training.num_train_epochs
        self.train_batch_size_per_device = config.training.train_batch_size
        self.gradient_accumulation_steps = config.training.gradient_accumulation_steps
        self.eval_size_ratio = config.dataset.eval_ratio
        self.total_save_limit = config.training.total_save_limit
        self.clear_memory_steps = config.training.clear_memory_steps
        self.memory_log_steps = config.training.memory_log_steps

        # Cálculos derivados
        self.dataset_size = None  # Será definido posteriormente
        self.num_chunks = len(config.training.mlm_probabilities)

    def calculate_derived_params(self, dataset_size: int):
        self.dataset_size = dataset_size
        self.chunk_size = self.dataset_size // self.num_chunks
        self.eval_size_per_chunk = int(self.chunk_size * self.eval_size_ratio)
        self.chunk_train_size = self.chunk_size - self.eval_size_per_chunk
        self.total_train_steps = (
            self.chunk_train_size // self.train_batch_size_per_device
        ) * self.num_train_epochs
        self.effective_batch_size = (
            self.train_batch_size_per_device * self.gradient_accumulation_steps
        )
        self.total_steps_per_epoch = math.ceil(
            self.dataset_size / self.effective_batch_size
        )


class TrainingComponents:
    def __init__(self, model, training_config, total_steps):
        self.model = model
        self.training_config = training_config
        self.total_steps = total_steps

    def setup_components(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )

        scaler = GradScaler(
            enabled=(self.model.device.type == "cuda" and self.training_config.fp16)
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.training_config.warmup_steps,
            num_training_steps=self.total_steps,
        )

        return optimizer, scaler, scheduler


def tokenize_function(examples, tokenizer, target_column="text"):
    """
    Função para tokenizar exemplos do dataset.

    Args:
        examples: Batch de exemplos a serem tokenizados
        tokenizer: Tokenizador a ser utilizado
        target_column: Nome da coluna contendo o texto

    Returns:
        Exemplos tokenizados
    """
    return tokenizer(
        examples[target_column],
        return_special_tokens_mask=True,
    )


def tokenize_dataset(dataset, tokenizer, num_proc=4):
    """
    Tokeniza o dataset completo usando processamento paralelo.

    Args:
        dataset: Dataset a ser tokenizado
        tokenizer: Tokenizador a ser utilizado
        num_proc: Número de processos para paralelização

    Returns:
        Dataset tokenizado
    """
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
    )

    return tokenized_dataset
