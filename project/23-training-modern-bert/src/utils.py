import gc
import logging

import torch
from tabulate import tabulate
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def log_gpu_memory_usage():
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        device_properties = torch.cuda.get_device_properties(device)
        total_memory = device_properties.total_memory / 1024**2
        allocated_memory = torch.cuda.memory_allocated(device) / 1024**2
        reserved_memory = torch.cuda.memory_reserved(device) / 1024**2
        allocated_percent = (allocated_memory / total_memory) * 100
        reserved_percent = (reserved_memory / total_memory) * 100
        table = tabulate(
            [
                ["Total de Memória (MB)", f"{int(round(total_memory))}", "100%"],
                [
                    "Memória Alocada (MB)",
                    f"{int(round(allocated_memory))}",
                    f"{int(round(allocated_percent))}%",
                ],
                [
                    "Memória Reservada (MB)",
                    f"{int(round(reserved_memory))}",
                    f"{int(round(reserved_percent))}%",
                ],
            ],
            headers=["Descrição", "Valor", "Percentual"],
            tablefmt="grid",
        )
        logger.info(f"\nUso de VRAM pela GPU:\n{table}")
    else:
        logger.info("CUDA não está disponível.")


def clear_vram():
    """Limpa a memória da GPU (VRAM)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    torch.cuda.empty_cache()

    gc.collect()
    if hasattr(torch.cuda, "ipc_collect"):
        torch.cuda.ipc_collect()
    gc.collect()
    logger.info("VRAM cleared definitively.")


def forward_pass(model, inputs, device):
    """
    Realiza uma passagem para frente no modelo.

    Args:
        model: Modelo para realizar a passagem
        inputs: Entradas do modelo
        device: Dispositivo onde o modelo está

    Returns:
        Perda calculada pelo modelo

    Raises:
        ValueError: Se o modelo não retornar uma perda
    """
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        outputs = model(**inputs, return_dict=True)
    if outputs.loss is None:
        raise ValueError("Model did not return a loss.")
    return outputs.loss


def evaluate(model, eval_dataset, data_collator, batch_size, device):
    """
    Avalia o desempenho do modelo no conjunto de validação.

    Args:
        model: Modelo a ser avaliado
        eval_dataset: Dataset de avaliação
        data_collator: Colator de dados para preparar batches
        batch_size: Tamanho do batch para avaliação
        device: Dispositivo onde o modelo está

    Returns:
        Perda média de avaliação
    """
    model.eval()
    losses = []
    eval_iterator = eval_dataset.iter(batch_size=batch_size)

    for batch in tqdm(eval_iterator, desc="Evaluating"):
        with (
            torch.no_grad(),
            torch.amp.autocast("cuda", enabled=(device.type == "cuda")),
        ):
            try:
                inputs = data_collator(batch)
                loss = forward_pass(model, inputs, device)
                losses.append(loss.item())
            except Exception as e:
                logger.warning(f"Evaluation batch failed: {e}. Skipping.")
                continue

    model.train()
    average_loss = sum(losses) / len(losses) if losses else float("inf")
    return average_loss
