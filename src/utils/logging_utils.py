import os
import torch
import wandb
import logging
from typing import Dict, Any, Optional, Union


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Setup a Python logger with the specified name and level.
    
    Args:
        name: Name for the logger
        level: Logging level (default: logging.INFO)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create console handler if no handlers exist
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def configure_wandb(
    project_name: str, 
    config: Dict[str, Any], 
    run_name: Optional[str] = None,
    tags: Optional[list] = None
) -> wandb.run:
    """
    Configure and initialize Weights & Biases for experiment tracking.
    
    Args:
        project_name: Name of the wandb project
        config: Configuration dictionary to be logged
        run_name: Optional name for this specific run
        tags: Optional list of tags for the run
        
    Returns:
        wandb.run: Initialized wandb run object
    """
    return wandb.init(
        project=project_name,
        name=run_name,
        config=config,
        tags=tags
    )


def log_metrics(
    logger: Union[wandb.run, logging.Logger],
    metrics: Dict[str, Any], 
    step: Optional[int] = None,
    commit: bool = True
) -> None:
    """
    Log metrics to the appropriate logger.
    
    Args:
        logger: Logger object (wandb.run or logging.Logger)
        metrics: Dictionary of metrics to log
        step: Optional step for wandb logging
        commit: Whether to commit the metrics immediately (wandb only)
    """
    if logger is None:
        return
        
    if isinstance(logger, type(wandb.run)):
        # Log to wandb
        logger.log(metrics, step=step, commit=commit)
    elif isinstance(logger, logging.Logger):
        # Log to Python logger
        metrics_str = ", ".join([f"{k}: {v}" for k, v in metrics.items()])
        if step is not None:
            logger.info(f"Step {step}: {metrics_str}")
        else:
            logger.info(f"Metrics: {metrics_str}")


class ModelLoggerHandler:
    """
    Handles model logging and saving operations during training.
    """

    def __init__(self, logger=None, model_name="model", save_freq=1, save_best=True):
        self.logger = logger
        self.model_name = model_name
        self.save_freq = save_freq
        self.save_best = save_best
        self.best_loss = float("inf")

        # Setup save directory if using wandb
        if isinstance(logger, type(wandb.run)):
            self.save_dir = os.path.join(wandb.run.dir, "checkpoints")
            os.makedirs(self.save_dir, exist_ok=True)
        else:
            self.save_dir = "."

    def setup(self, model, config=None):
        """Initialize logger with config and model watching."""
        if self.logger is not None and config is not None:
            if hasattr(self.logger, "config"):
                for key, value in config.items():
                    self.logger.config[key] = value

            if hasattr(self.logger, "watch"):
                self.logger.watch(model, log_freq=100)

    def log_batch(
        self, loss, batch_idx, epoch, total_batches, tag="train", loss_components=None
    ):
        """Log batch-level metrics with optional individual loss components.

        Args:
            loss: The total loss value
            batch_idx: Current batch index
            epoch: Current epoch
            total_batches: Total number of batches per epoch
            tag: Tag to differentiate train/val metrics
            loss_components: Dictionary of individual loss components to log
        """
        if self.logger is not None and hasattr(self.logger, "log"):
            metrics = {
                f"{tag} batch_loss": loss.item(),
                f"{tag} batch": batch_idx + epoch * total_batches,
            }

            # Add individual loss components if provided
            if loss_components is not None:
                for name, value in loss_components.items():
                    metrics[f"{tag} {name}"] = (
                        value.item() if torch.is_tensor(value) else value
                    )

            self.logger.log(metrics)

    def log_epoch(
        self, epoch, avg_loss, avg_val_loss, learning_rate, loss_component_avgs=None
    ):
        """Log epoch-level metrics with optional individual loss component averages.

        Args:
            epoch: Current epoch
            avg_loss: Average total loss for the epoch
            avg_val_loss: Average validation loss for the epoch
            learning_rate: Current learning rate
            loss_component_avgs: Dictionary of average individual loss components
        """
        if self.logger is not None and hasattr(self.logger, "log"):
            metrics = {
                "epoch": epoch + 1,
                "epoch_loss": avg_loss,
                "learning_rate": learning_rate,
                "avg_val_loss": avg_val_loss,
            }

            # Add individual loss component averages if provided
            if loss_component_avgs is not None:
                for name, value in loss_component_avgs.items():
                    metrics[f"avg_{name}"] = value

            self.logger.log(metrics)

    def save_checkpoint(self, epoch, model, optimizer, loss, is_best=False):
        """Save model checkpoint and upload to wandb if available."""
        if self.logger is None:
            return

        # Determine checkpoint name
        checkpoint_name = (
            f"{self.model_name}_best.pt"
            if is_best
            else f"{self.model_name}_epoch_{epoch+1}.pt"
        )
        checkpoint_path = os.path.join(self.save_dir, checkpoint_name)

        # Save checkpoint locally
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                # 'optimizer_state_dict': optimizer.state_dict(),
                "loss": loss,
            },
            checkpoint_path,
        )

        # Upload to wandb if available
        # if isinstance(self.logger, type(wandb.run)):
        #     artifact = wandb.Artifact(
        #         name=f'{self.model_name}_{"best" if is_best else "checkpoint"}',
        #         type='model',
        #         description=f'{"Best model" if is_best else "Model"} checkpoint from epoch {epoch+1}'
        #     )
        # artifact.add_file(checkpoint_path)
        # self.logger.log_artifact(artifact)
        # Clean up local checkpoint
        # os.remove(checkpoint_name)

    def handle_epoch_end(self, epoch, model, optimizer, avg_loss):
        """Handle end of epoch operations (logging and saving)."""
        # Save based on frequency
        # if (epoch + 1) % self.save_freq == 0:
        #     self.save_checkpoint(epoch, model, optimizer, avg_loss)

        # Save best model if enabled
        if self.save_best and avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self.save_checkpoint(epoch, model, optimizer, avg_loss, is_best=True)

    def finish(self):
        """Cleanup logging resources."""
        if self.logger is not None and hasattr(self.logger, "finish"):
            self.logger.finish()