import logging
import yaml
from pydantic import BaseModel, Field


def set_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("logs/training.log"), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    return logger


class ModelConfig(BaseModel):
    base_id: str = Field(..., description="ID do modelo base no Hugging Face Hub")
    tokenizer_path: str = Field(..., description="Caminho para o tokenizador")
    output_dir: str = Field(..., description="Diretório para salvar o modelo treinado")


class DatasetConfig(BaseModel):
    id: str = Field(..., description="ID do dataset no HF Hub")
    max_news: int = Field(..., description="Número máximo de notícias a usar")
    max_sentences: int = Field(..., description="Número máximo de sentenças")
    eval_ratio: float = Field(..., description="Fração dos dados usada para validação")


class TrainingConfig(BaseModel):
    num_train_epochs: int = Field(..., description="Número de épocas de treinamento")
    train_batch_size: int = Field(..., description="Tamanho do batch por dispositivo")
    gradient_accumulation_steps: int = Field(
        ..., description="Passos de acumulação de gradiente"
    )
    learning_rate: float = Field(..., description="Taxa de aprendizado inicial")
    weight_decay: float = Field(..., description="Peso de decaimento")
    warmup_steps: int = Field(..., description="Passos de aquecimento para o scheduler")
    mlm_probabilities: list[float] = Field(
        ..., description="Probabilidades de mascaramento para diferentes chunks"
    )
    total_save_limit: int = Field(
        ..., description="Número máximo de checkpoints a manter"
    )
    fp16: bool = Field(..., description="Usar treinamento em precisão mista")
    use_flash_attention: bool = Field(
        ..., description="Usar Flash Attention se disponível"
    )
    push_interval: int = Field(
        ..., description="Intervalo de steps para push para o Hub"
    )
    num_workers: int = Field(
        ..., description="Número de workers para processamento de dados"
    )
    clear_memory_steps: int = Field(
        ..., description="Número de passos antes de limpar a memória"
    )
    memory_log_steps: int = Field(
        ..., description="Número de passos para retornar um log de uso da memória"
    )


class ExecutionConfig(BaseModel):
    testing: bool = Field(..., description="Executar em modo de teste")
    seed: int = Field(..., description="Semente para reprodutibilidade")
    device: str = Field(..., description="Dispositivo de execução: auto, cuda ou cpu")


class MLMTrainingConfig(BaseModel):
    model: ModelConfig
    dataset: DatasetConfig
    training: TrainingConfig
    execution: ExecutionConfig


def parse_yaml(file_path: str) -> MLMTrainingConfig:
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return MLMTrainingConfig(**data)
