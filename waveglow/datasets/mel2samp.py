# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************\
import inspect
import pathlib
import random
from typing import List, Optional, Tuple, Dict, Any

import tacotron2.hparams as taco_hparams
import tacotron2.models._layers as taco_layers
import torch
import torch.utils.data
from scipy.io.wavfile import read
from tacotron2 import factory
from tacotron2 import hparams as _hparams
from tacotron2.audio_preprocessors import _audio_preprocessor as audio_preprocessor

MAX_WAV_VALUE = 32768.0


def prepare_dataloaders(hparams: _hparams.HParams) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Returns train and valid dataloaders tuple.

    Args:
        hparams: Hyper parameters object

    Returns:
        tuple with train and valid data loaders
    """

    train_dataloader = Mel2Samp.from_hparams(hparams=hparams, is_valid=False).get_data_loader(
        batch_size=hparams.batch_size, shuffle=True
    )
    valid_dataloader = Mel2Samp.from_hparams(hparams=hparams, is_valid=True).get_data_loader(
        batch_size=hparams.batch_size, shuffle=False
    )

    return train_dataloader, valid_dataloader


def files_to_list(file_path: pathlib.Path, separator: Optional[str] = '|') -> List[pathlib.Path]:
    """Returns list of file path, which are listed in input file (one file path at a line).

    Args:
        file_path: Path to the input file.
        separator: Each line from the input file will be split at this separator. The first (0 index) item will
            be treated as an audio file path. If None, lines will note be split (the whole line is an audio file path)

    Returns: list of file paths

    """
    from_dir = file_path.parent
    with file_path.open() as f:
        if separator is not None:
            file_lines = [line.split('|')[0].strip() for line in f.readlines()]
        else:
            file_lines = [line.strip() for line in f.readlines()]

        file_paths = [from_dir / file_path for file_path in file_lines]

    return file_paths


def load_wav_to_torch(full_path, audio_preprocessors: List[audio_preprocessor.AudioPreprocessor]) \
        -> Tuple[torch.tensor, int]:
    """Load audio file, apply audio preprocessors and transform to torch tensor.

    Args:
        full_path: Path to the audio file.
        audio_preprocessors: List with audio preprocessor objects (from tacotron2 library).

    Returns:
        Tuple with audio tensor and it's sampling rate
    """
    sampling_rate, audio = read(full_path)

    for preprocessor in audio_preprocessors:
        audio = preprocessor(audio)

    return torch.from_numpy(audio).float(), sampling_rate


class Mel2Samp(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """

    def __init__(self, meta_file_path, segment_length, filter_length, hop_length, win_length, sampling_rate, mel_fmin,
                 mel_fmax, n_mel_channels: int, audio_preprocessors: List[audio_preprocessor.AudioPreprocessor]):
        self.audio_files = files_to_list(meta_file_path)

        self.audio_preprocessors = audio_preprocessors

        self.stft: taco_layers.TacotronSTFT = taco_layers.TacotronSTFT(
            filter_length=filter_length,
            hop_length=hop_length,
            win_length=win_length,
            sampling_rate=sampling_rate,
            mel_fmin=mel_fmin,
            mel_fmax=mel_fmax,
            n_mel_channels=n_mel_channels
        )
        self.segment_length = segment_length
        self.sampling_rate = sampling_rate

    @classmethod
    def from_hparams(cls, hparams: taco_hparams.HParams, is_valid: bool):
        """Build class instance from hparams map
        If you create dataset instance via this method, make sure, that meta_train.txt (if is_valid==False) or
            meta_valid.txt (is is_valid==True) exists in the dataset directory
        :param hparams: HParams, dictionary with parameters
        :param is_valid: bool, get validation dataset or not (train)
        :return: TextMelLoader, dataset instance
        """
        param_names = inspect.getfullargspec(cls.__init__).args
        params = dict()
        for param_name in param_names:
            param_value = cls._get_param_value(param_name=param_name, hparams=hparams, is_valid=is_valid)

            if param_value is not None:
                params[param_name] = param_value

        obj = cls(**params)
        return obj

    @staticmethod
    def _get_param_value(param_name: str, hparams: taco_hparams.HParams, is_valid: bool) -> Any:
        if param_name == 'self':
            value = None
        elif param_name == 'meta_file_path':
            data_directory = pathlib.Path(hparams.data_directory)
            postfix = 'valid' if is_valid else 'train'
            value = data_directory / f'meta_{postfix}.txt'
            if not value.is_file():
                raise FileNotFoundError(f"Can't find {str(value)} file. Make sure, that file exists")
        elif param_name == 'audio_preprocessors':
            value = [
                factory.Factory.get_object(f'tacotron2.audio_preprocessors.{k}', **v)
                for k, v in hparams.audio_preprocessors.items()
            ]
        else:
            value = hparams[param_name]

        return value

    def get_mel(self, audio):
        audio_norm = audio / MAX_WAV_VALUE
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        return melspec

    def __getitem__(self, index):
        # Read audio
        filename = self.audio_files[index]
        audio, sampling_rate = load_wav_to_torch(filename, audio_preprocessors=self.audio_preprocessors)

        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))

        # Take segment
        if audio.size(0) >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start:audio_start + self.segment_length]
        else:
            audio = torch.nn.functional.pad(audio, (0, self.segment_length - audio.size(0)), 'constant').data

        mel = self.get_mel(audio)
        audio = audio / MAX_WAV_VALUE

        return mel, audio

    def get_data_loader(self, batch_size: int, shuffle: bool) -> torch.utils.data.DataLoader:
        """Constructs DataLoader object from the Dataset object.

        Args:
            batch_size: Training (or validation) batch size.
            shuffle: Set True if you want to shuffle the data (will be set False in case of distributed training).

        Returns:
            Prepared dataloader object.
        """

        dataloader = torch.utils.data.DataLoader(
            self,
            num_workers=1,
            batch_size=batch_size,
            pin_memory=False,
            drop_last=True,
            shuffle=shuffle,
            collate_fn=Mel2SampCollate()
        )

        return dataloader

    def __len__(self):
        return len(self.audio_files)


class Mel2SampCollate:
    """Class-caller which represents a collate function for Mel2Samp dataset"""

    def __call__(self, batch) -> Dict:
        """Collates training batch with mel and audio.

        Args:
            batch: Batch with mel spectrogram and audio.

        Returns:
            Collated batch dictionary (with x and y field)
        """

        mel, audio = list(zip(*batch))

        audio = torch.stack(audio)
        mel = torch.stack(mel)

        batch = {'x': mel, 'y': audio}

        return batch
