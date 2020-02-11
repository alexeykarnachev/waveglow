from tacotron2 import learner
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

    def on_opt_step(self, learner: learner.Learner):
        pass
