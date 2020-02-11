from tacotron2 import learner as taco_learner
from tacotron2.callbacks import _callback as callback
from torch.utils import tensorboard


class TensorBoardLoggerCallback(callback.Callback):
    """Tensorboard writer callback"""

    def __init__(self, summary_writer: tensorboard.SummaryWriter):
        """
        Args:
            summary_writer: tensroboard summary writer object
        """
        self.summary_writer = summary_writer

    def on_opt_step(self, learner: taco_learner.Learner):
        """Writes training loss, grad. norm and learning rate at each optimizer step.

        Args:
            learner: tacotron2 learner object.

        """
        learning_rate = learner.optimizer.param_groups[0]['lr']

        self.summary_writer.add_scalar("training.loss", learner.train_loss, learner.overall_step)
        self.summary_writer.add_scalar("grad.norm", learner.grad_norm, learner.overall_step)
        self.summary_writer.add_scalar("learning.rate", learning_rate, learner.overall_step)

    def on_eval_end(self, learner: taco_learner.Learner):
        """Writes validation loss at each validation step end.

        Args:
            learner: tacotron2 learner object

        """
        self.summary_writer.add_scalar("validation.loss", learner.valid_loss, learner.overall_step)
