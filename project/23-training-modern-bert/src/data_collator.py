from typing import Any

import torch
from transformers import DataCollatorForLanguageModeling


class DynamicPaddingDataCollator(DataCollatorForLanguageModeling):
    """
    Colator de dados com preenchimento dinâmico para MLM.
    """

    def __call__(self, examples: dict[str, Any]) -> dict[str, torch.Tensor]:
        # Encontra o comprimento máximo dentro do batch atual
        max_length = max(len(input_ids) for input_ids in examples["input_ids"])

        # Preenche ou trunca cada exemplo para o compriment o máximo
        batch = []
        input_ids = examples["input_ids"]
        attention_mask = examples["attention_mask"]

        for ids, mask in zip(input_ids, attention_mask):
            padding_length = max_length - len(ids)
            if padding_length > 0:
                # Preenche
                ids = torch.tensor(ids + [self.tokenizer.pad_token_id] * padding_length)
                mask = torch.tensor(mask + [0] * padding_length)
            elif padding_length <= 0:
                # Trunca (se habilitado no tokenizador)
                ids = torch.tensor(ids[:max_length])
                mask = torch.tensor(mask[:max_length])

            batch.append({"input_ids": ids, "attention_mask": mask})

        # Aplica a lógica restante do colator (mascaramento MLM, etc.)
        batch = self.torch_call(batch)

        # Garante formas e tipos corretos
        batch = self.fix_batch_inputs(batch)

        return batch

    def fix_batch_inputs(self, inputs: dict) -> dict:
        """
        Garante que os tensores de entrada tenham a forma e o tipo corretos.

        Args:
            inputs: Dicionário com tensores de entrada

        Returns:
            Dicionário com tensores corrigidos

        Raises:
            ValueError: Se algum tensor tiver forma inesperada
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
