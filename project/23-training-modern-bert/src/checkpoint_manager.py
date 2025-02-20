import os
import shutil

import torch

from .config import set_logger

logger = set_logger()


class CheckpointManager:
    def __init__(self, output_dir, model, optimizer, scheduler, total_save_limit: int = 2):
        self.output_dir = output_dir
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.total_save_limit = total_save_limit

    def save_checkpoint(self, global_step, epoch, tokenizer):
        checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{global_step}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        torch.save(
            {
                "global_step": global_step,
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            os.path.join(checkpoint_dir, "training_state.pt"),
        )

        self.model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        logger.info(f"Checkpoint saved at {checkpoint_dir}")

        self._cleanup_old_checkpoints()

    def load_latest_checkpoint(self):
        if not os.path.exists(self.output_dir):
            return 0, 0

        checkpoints = [
            d for d in os.listdir(self.output_dir) if d.startswith("checkpoint-")
        ]
        if not checkpoints:
            return 0, 0

        latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
        checkpoint_path = os.path.join(self.output_dir, latest_checkpoint)

        checkpoint = torch.load(os.path.join(checkpoint_path, "training_state.pt"))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return checkpoint["global_step"], checkpoint["epoch"]

    def _cleanup_old_checkpoints(self):
        checkpoints = [
            d for d in os.listdir(self.output_dir) if d.startswith("checkpoint-")
        ]
        if len(checkpoints) > self.total_save_limit:
            checkpoints_to_delete = sorted(
                checkpoints, key=lambda x: int(x.split("-")[1])
            )[: -self.total_save_limit]
            for checkpoint in checkpoints_to_delete:
                shutil.rmtree(os.path.join(self.output_dir, checkpoint))
                logger.info(f"Deleted old checkpoint: {checkpoint}")
