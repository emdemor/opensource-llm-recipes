from .config import parse_yaml, set_logger
from .data_preprocessing import DataPreprocessor
from .model_manager import ModelManager
from .trainer import Trainer

logger = set_logger()


def run(config_path, from_checkpoint=True):
    # Carrega a configuração
    logger.info(f"Read config from {config_path}")
    config = parse_yaml(config_path)

    # Pré-processamento dos dados
    data_preprocessor = DataPreprocessor(config.dataset)
    dataset = data_preprocessor.load_and_prepare_data()

    # Inicialização do modelo e tokenizador
    model_manager = ModelManager(config.model, config.training)
    model, tokenizer = model_manager.initialize_model()

    # Treinamento
    trainer = Trainer(config, model, tokenizer)
    trainer.train(dataset, from_checkpoint=from_checkpoint)
