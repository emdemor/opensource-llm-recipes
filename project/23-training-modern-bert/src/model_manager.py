import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

from .config import set_logger

logger = set_logger()


class ModelManager:
    def __init__(self, model_config, training_config):
        self.model_config = model_config
        self.training_config = training_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize_model(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.tokenizer_path, clean_up_tokenization_spaces=False
        )

        model_config = AutoConfig.from_pretrained(self.model_config.base_id)
        if self.training_config.fp16:
            model_config.torch_dtype = torch.float16

        model = AutoModelForMaskedLM.from_pretrained(
            self.model_config.base_id, config=model_config
        )
        model.resize_token_embeddings(len(tokenizer))
        model.to(self.device)

        self.set_attention(model)
        self.check_vocab_size(tokenizer, model)

        return model, tokenizer

    def set_attention(self, model):
        if not self.training_config.use_flash_attention:
            return model

        if not torch.cuda.is_available():
            return model

        try:
            from flash_attn import flash_attn_qkvpacked_func

            qkv = torch.randn(1, 1, 3, 16, 64, dtype=torch.float16, device="cuda")
            flash_attn_qkvpacked_func(qkv, causal=False)
            logger.info("Flash Attention is compatible.")
        except (ImportError, RuntimeError) as e:
            logger.warning(f"Flash Attention is not compatible: {str(e)}")
            return model

        try:
            from flash_attn import FlashAttention

            for module in model.modules():
                if isinstance(module, nn.MultiheadAttention):
                    module.attention = FlashAttention()
            logger.info("FlashAttention integrated successfully.")
        except Exception as e:
            logger.error(f"Failed to integrate FlashAttention: {str(e)}")

        return model

    def check_vocab_size(self, tokenizer, model):
        max_token_id = max(tokenizer.get_vocab().values())
        logger.info(f"Maior ID no tokenizador: {max_token_id}")
        logger.info(f"Tamanho do vocabulário do modelo: {model.config.vocab_size}")
        assert (
            max_token_id < model.config.vocab_size
        ), "IDs de tokens fora do intervalo!"
