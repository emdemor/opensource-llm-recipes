from collections import defaultdict

import torch
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

from .config import set_logger
from .checkpoint_manager import CheckpointManager
from .data_collator import DynamicPaddingDataCollator
from .training_components import TrainingParams, TrainingComponents, tokenize_dataset
from .utils import log_gpu_memory_usage

logger = set_logger()


class Trainer:
    def __init__(self, config, model, tokenizer):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        self.global_step = 0
        self.start_epoch = 0
        self.training_history = defaultdict(list)

    def train(self, dataset, from_checkpoint=True):
        training_params = TrainingParams(self.config)
        training_params.calculate_derived_params(len(dataset))

        optimizer, scaler, scheduler = TrainingComponents(
            self.model, self.config.training, training_params.total_train_steps
        ).setup_components()

        checkpoint_manager = CheckpointManager(
            self.config.model.output_dir,
            self.model,
            optimizer,
            scheduler,
            training_params.total_save_limit,
        )

        if from_checkpoint:
            self.global_step, self.start_epoch = (
                checkpoint_manager.load_latest_checkpoint()
            )
        else:
            self.global_step, self.start_epoch = (0, 0)

        logger.info(
            f"Starting training with global_step = {self.global_step} | epoch = {self.start_epoch}"
        )

        try:
            for epoch in range(self.start_epoch, training_params.num_train_epochs):
                logger.info(
                    f"Starting epoch {1+epoch}/{training_params.num_train_epochs}"
                )
                self._train_epoch(
                    epoch,
                    training_params,
                    optimizer,
                    scaler,
                    scheduler,
                    checkpoint_manager,
                    dataset,
                )

            self._save_final_model()

        except KeyboardInterrupt:
            logger.info("Training interrupted. Saving final checkpoint...")
            checkpoint_manager.save_checkpoint(self.global_step, epoch, self.tokenizer)

    def plot_training_progress(self, save_path=None):
        """
        Plota a evolução da loss de treinamento e avaliação ao longo dos steps.

        Args:
            save_path (str, optional): Caminho para salvar o gráfico.
                                    Se None, exibe o gráfico interativamente.
        """
        plt.figure(figsize=(12, 6))

        # Plotando loss de treinamento
        plt.plot(
            self.training_history["steps"],
            self.training_history["train_loss"],
            label="Training Loss",
            alpha=0.5,
            color="blue",
        )

        # Plotando loss de avaliação
        if len(self.training_history["eval_steps"]) > 0:
            plt.plot(
                self.training_history["eval_steps"],
                self.training_history["eval_loss"],
                label="Evaluation Loss",
                marker="o",
                linestyle="--",
                color="red",
            )

        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.title("Training and Evaluation Loss Over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def save_training_history(self, filepath):
        """
        Salva o histórico de treinamento em um arquivo para análise posterior.
        
        Args:
            filepath (str): Caminho onde o arquivo será salvo
        """
        import json
        
        history_to_save = {
            k: [float(v) for v in vals] 
            for k, vals in self.training_history.items()
        }
        
        with open(filepath, 'w') as f:
            json.dump(history_to_save, f)

    def _train_epoch(
        self, epoch, params, optimizer, scaler, scheduler, checkpoint_manager, dataset
    ):
        tokenized_dataset = tokenize_dataset(
            dataset, self.tokenizer, num_proc=self.config.training.num_workers
        )

        for chunk_number, mlm_probability in enumerate(
            self.config.training.mlm_probabilities
        ):
            logger.info(
                f"Starting chunk {1+chunk_number}/{params.num_chunks} | MLM Prob = {mlm_probability}"
            )
            data_collator = DynamicPaddingDataCollator(
                self.tokenizer, mlm_probability=mlm_probability
            )

            train_dataset, eval_dataset = self._prepare_datachunk(
                chunk_number, params, tokenized_dataset
            )

            self._train_chunk(
                epoch,
                chunk_number,
                mlm_probability,
                params,
                train_dataset,
                eval_dataset,
                data_collator,
                optimizer,
                scaler,
                scheduler,
                checkpoint_manager,
            )

    def _prepare_datachunk(self, chunk_number, params, tokenized_dataset):
        eval_start_idx = chunk_number * params.chunk_size
        eval_end_idx = eval_start_idx + params.eval_size_per_chunk - 1
        train_start_idx = chunk_number * params.chunk_size + params.eval_size_per_chunk
        train_end_idx = train_start_idx + params.chunk_train_size - 1

        logger.info(
            f"Splitting | "
            f"chunk: {eval_start_idx}-{train_end_idx} | "
            f"eval: {eval_start_idx}-{eval_end_idx} | "
            f"train: {train_start_idx}-{train_end_idx}"
        )

        train_dataset = (
            tokenized_dataset.skip(train_start_idx)
            .take(params.chunk_train_size)
            .shuffle(seed=42)
        )

        eval_dataset = (
            tokenized_dataset.skip(eval_start_idx)
            .take(params.eval_size_per_chunk)
            .shuffle(seed=42)
        )

        return train_dataset, eval_dataset

    def _train_chunk(
        self,
        epoch,
        chunk_number,
        mlm_probability,
        params,
        train_dataset,
        eval_dataset,
        data_collator,
        optimizer,
        scaler,
        scheduler,
        checkpoint_manager,
    ):
        logger.info(
            f"Epoch {epoch + 1}/{params.num_train_epochs} | MLM Probability: {mlm_probability}"
        )

        train_iterator = train_dataset.iter(
            batch_size=params.train_batch_size_per_device
        )

        for step, batch in tqdm(
            enumerate(train_iterator), desc=f"Training (MLM {mlm_probability})"
        ):
            accumulation_step_complete = (
                step + 1
            ) % params.gradient_accumulation_steps == 0

            try:
                self._training_step(batch, data_collator, scaler)

                if accumulation_step_complete:
                    self._update_model(optimizer, scaler, scheduler)
                    self.global_step += 1

                    self._periodic_operations(
                        params,
                        eval_dataset,
                        data_collator,
                        optimizer,
                        scaler,
                        scheduler,
                        checkpoint_manager,
                    )

            except Exception as e:
                logger.error(f"Training batch failed: {e}. Skipping.")
                continue

        self._evaluate_chunk(chunk_number, eval_dataset, data_collator, params)
        logger.info(f"Saving checkpoint after chunk {chunk_number}.")
        checkpoint_manager.save_checkpoint(self.global_step, epoch, self.tokenizer)

    def _training_step(self, batch, data_collator, scaler):
        inputs = data_collator(batch)
        loss = forward_pass(self.model, inputs, self.device)
        scaler.scale(loss / self.config.training.gradient_accumulation_steps).backward()

        current_loss = loss.item()
        self.training_history["train_loss"].append(current_loss)
        self.training_history["steps"].append(self.global_step)

        return current_loss

    def _update_model(self, optimizer, scaler, scheduler):
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()

    def _periodic_operations(
        self,
        params,
        eval_dataset,
        data_collator,
        optimizer,
        scaler,
        scheduler,
        checkpoint_manager,
    ):
        eval_interval = max(1, params.total_steps_per_epoch // 4)
        if self.global_step % eval_interval == 0:
            eval_loss = evaluate(
                self.model,
                eval_dataset,
                data_collator,
                batch_size=params.train_batch_size_per_device,
                device=self.device,
            )
            self.training_history["eval_loss"].append(eval_loss)
            self.training_history["eval_steps"].append(self.global_step)
            logger.info(f"Evaluation loss at step {self.global_step}: {eval_loss}")
            self.plot_training_progress()

        if self.global_step % self.config.training.push_interval == 0:
            checkpoint_manager.save_checkpoint(
                self.global_step, self.start_epoch, self.tokenizer
            )

        if self.global_step % params.memory_log_steps == 0:
            log_gpu_memory_usage()

        if (
            self.device.type == "cuda"
            and self.global_step % params.clear_memory_steps == 0
        ):
            torch.cuda.empty_cache()

    def _evaluate_chunk(self, chunk_number, eval_dataset, data_collator, params):
        logger.info(f"Evaluating at the end of chunk {chunk_number}...")
        eval_loss = evaluate(
            self.model,
            eval_dataset,
            data_collator,
            batch_size=params.train_batch_size_per_device,
            device=self.device,
        )
        logger.info(f"Chunk evaluation loss: {eval_loss}")

    def _save_final_model(self):
        logger.info("Training complete. Saving final model...")
        self.model.save_pretrained(self.config.model.output_dir)
        self.tokenizer.save_pretrained(self.config.model.output_dir)


def fix_batch_inputs(inputs: dict) -> dict:
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
    inputs = fix_batch_inputs(inputs)
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
        with torch.no_grad(), torch.amp.autocast(
            "cuda", enabled=(device.type == "cuda")
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
