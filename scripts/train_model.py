import argparse
import shutil
import warnings
from pathlib import Path

from tacotron2 import factory
from tacotron2 import hparams
from tacotron2 import learner
from tacotron2 import utils as taco_utils
from tacotron2.callbacks import model_save_callback
from tacotron2.callbacks import reduce_lr_on_plateau_callback
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW

from waveglow.callbacks import callbacks
from waveglow.datasets import mel2samp

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Run BERT based Bi-Encoder experiment')

parser.add_argument(
    '--experiments_dir', type=Path, required=True, help='Root directory of all your experiments'
)
parser.add_argument(
    '--hparams_file', type=Path, required=True, help='Path to the hparams yaml file'
)
parser.add_argument(
    '--tb_logdir', type=Path, required=True, help='Tensorboard logs directory'
)

args = parser.parse_args()

hparams = hparams.HParams.from_yaml(args.hparams_file)
experiments_dir = args.experiments_dir
experiment_id = taco_utils.get_cur_time_str()
tb_logdir = args.tb_logdir / experiment_id

experiment_dir: Path = experiments_dir / experiment_id
experiment_dir.mkdir(exist_ok=False, parents=True)
shutil.copy(str(args.hparams_file), str(experiment_dir / 'hparams.yaml'))
taco_utils.dump_json(args.__dict__, experiment_dir / 'arguments.json')
models_dir = experiment_dir / 'models'

if __name__ == '__main__':
    taco_utils.seed_everything(hparams.seed)
    dl = mel2samp.prepare_dataloaders(hparams)

    model = factory.Factory.get_class(f'waveglow.models.{hparams.model_class_name}')(hparams).to(hparams.device)
    optimizer = AdamW(model.parameters(), lr=hparams.learning_rate, weight_decay=hparams.weight_decay or 0)

    summary_writer = SummaryWriter(log_dir=tb_logdir)

    callbacks = [
        callbacks.TensorBoardLoggerCallback(
            summary_writer=summary_writer
        ),
        model_save_callback.ModelSaveCallback(
            hold_n_models=3,
            models_dir=models_dir
        )
    ]

    if None not in (hparams.lr_reduce_patience, hparams.lr_reduce_factor):
        callbacks.append(
            reduce_lr_on_plateau_callback.ReduceLROnPlateauCallback(
                patience=hparams.lr_reduce_patience,
                reduce_factor=hparams.lr_reduce_factor
            )
        )

    learner = learner.Learner(
        model=model,
        optimizer=optimizer,
        callbacks=callbacks
    ).fit(
        dl=dl,
        n_epochs=hparams.epochs,
        device=hparams.device,
        accum_steps=hparams.accum_steps,
        eval_steps=hparams.iters_per_checkpoint,
        use_all_gpu=hparams.use_all_gpu,
        fp16_opt_level=hparams.fp16_opt_level,
        max_grad_norm=hparams.grad_clip_thresh
    )
