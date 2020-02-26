import pathlib
import re

import numpy as np
import torch
from torch.utils import tensorboard

from waveglow import learner as lrn
from waveglow.callbacks import _callback


class TensorBoardLoggerCallback(_callback.Callback):
    """Tensorboard writer callback"""

    def __init__(self, summary_writer: tensorboard.SummaryWriter):
        """
        Args:
            summary_writer: tensroboard summary writer object
        """
        self.summary_writer = summary_writer

    def on_opt_step(self, learner: lrn.Learner):
        """Writes training loss, grad. norm and learning rate at each optimizer step.

        Args:
            learner: tacotron2 learner object.

        """
        learning_rate = learner.optimizer.param_groups[0]['lr']

        self.summary_writer.add_scalar("training.loss", learner.train_loss, learner.overall_step)
        self.summary_writer.add_scalar("grad.norm", learner.grad_norm, learner.overall_step)
        self.summary_writer.add_scalar("learning.rate", learning_rate, learner.overall_step)

    def on_eval_end(self, learner: lrn.Learner):
        """Writes validation loss at each validation step end.

        Args:
            learner: tacotron2 learner object

        """
        self.summary_writer.add_scalar("validation.loss", learner.valid_loss, learner.overall_step)


class ModelSaveCallback(_callback.Callback):
    def __init__(self, hold_n_models: int, models_dir: pathlib.Path, ):
        """
        Callback which saves the model
        :param hold_n_models: int, how many models to store (old models will be overwritten by the new ones)
        :param models_dir: output models directory
        """
        self.hold_n_models = hold_n_models
        self.models_dir = pathlib.Path(models_dir)

        self.models_dir.mkdir(exist_ok=False, parents=True)

    def on_eval_end(self, learner: lrn.Learner):
        self._save(learner)

    def on_train_end(self, learner):
        self._save(learner)

    def _save(self, learner: lrn.Learner):
        self._remove_old_models(models_dir=self.models_dir, hold_n_models=self.hold_n_models)
        self._save_model(learner, models_dir=self.models_dir)

    @staticmethod
    def _remove_old_models(models_dir: pathlib.Path, hold_n_models: int):
        file_paths = [path for path in list(models_dir.iterdir()) if re.match(r'model_\d+.pth', path.name)]

        if len(file_paths) >= hold_n_models:
            model_steps = [int(re.findall(r'\d+', file_path.name)[0]) for file_path in file_paths]
            sorted_file_path_ids = np.argsort(model_steps)
            n_models_to_remove = (len(file_paths) - hold_n_models) + 1
            for file_path_id in sorted_file_path_ids[:n_models_to_remove]:
                file_path = file_paths[file_path_id]
                file_path.unlink()

    @staticmethod
    def _save_model(learner: lrn.Learner, models_dir: pathlib.Path):
        save_dict = {
            "model_state_dict": learner.model.state_dict(),
            "optimizer_state_dict": learner.optimizer.state_dict(),
            "overall_step": learner.overall_step,
            "n_epochs": learner.n_epochs,
            "cur_epoch": learner.cur_epoch,
            "n_epoch_steps": learner.n_epoch_steps,
            "valid_loss": learner.valid_loss,
            "train_loss": learner.train_loss
        }
        model_file = models_dir / f'model_{learner.overall_step}.pth'
        torch.save(save_dict, str(model_file))


class ReduceLROnPlateauCallback(_callback.Callback):
    """Callback which reduces learning rate on loss plateau"""

    def __init__(self, patience: int, reduce_factor: float):
        """
        :param patience: int, number of epochs without loss improvements before lr reduce
        :param reduce_factor: float, multiplier to apply to learning rate on reduction
        """
        self.patience = patience
        self.reduce_factor = reduce_factor
        self.min_loss = np.inf
        self.counter = 0
        self.prev_reduce_loss = -np.inf

    def on_epoch_end(self, learner: lrn.Learner):
        cur_loss = learner.eval(learner.valid_dl)

        if np.isfinite(cur_loss):
            if cur_loss < self.min_loss:
                self.min_loss = cur_loss
                self.counter = 0
            else:
                self.counter += 1

            if self.counter >= self.patience and cur_loss >= self.prev_reduce_loss:
                self.prev_reduce_loss = cur_loss
                for g in learner.optimizer.param_groups:
                    g['lr'] *= self.reduce_factor

                self.counter = 0
