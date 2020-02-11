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
import pathlib
import random
from typing import List, Optional, Tuple

import tacotron2.models._layers as taco_layers
import torch
import torch.utils.data
from scipy.io.wavfile import read
from tacotron2 import hparams as _hparams

MAX_WAV_VALUE = 32768.0


def prepare_dataloaders(hparams: _hparams.HParams) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Returns train and valid dataloaders tuple.

    Args:
        hparams: Hyper parameters object

    Returns:
        tuple with train and valid data loaders
    """
    data_directory = pathlib.Path(hparams.data_directory)

    def _get_dataloader(training_files: pathlib.Path):
        dataset = Mel2Samp(
            training_files=training_files, segment_length=hparams.segment_length, filter_length=hparams.filter_length,
            hop_length=hparams.hop_length, win_length=hparams.win_length, sampling_rate=hparams.sampling_rate,
            mel_fmin=hparams.mel_fmin, mel_fmax=hparams.mel_fmax, n_mel_channels=hparams.n_mel_channels
        )
        dataloader = dataset.get_data_loader(
            batch_size=hparams.batch_size, is_distributed=hparams.use_all_gpu, shuffle=True
        )

        return dataloader

    train_dataloader = _get_dataloader(data_directory / 'meta_train.txt')
    valid_dataloader = _get_dataloader(data_directory / 'meta_valid.txt')

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


def load_wav_to_torch(full_path):
    """
    Loads wavdata into torch array
    """
    sampling_rate, data = read(full_path)
    return torch.from_numpy(data).float(), sampling_rate


class Mel2Samp(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """

    def __init__(self, training_files, segment_length, filter_length,
                 hop_length, win_length, sampling_rate, mel_fmin, mel_fmax, n_mel_channels: int):
        self.audio_files = files_to_list(training_files)

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
        audio, sampling_rate = load_wav_to_torch(filename)
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

        return (mel, audio)

    def get_data_loader(self, batch_size: int, is_distributed: bool, shuffle: bool) -> torch.utils.data.DataLoader:
        """Constructs DataLoader object from the Dataset object.

        Args:
            batch_size: Training (or validation) batch size.
            is_distributed: Set True, if you use distributed training.
            shuffle: Set True if you want to shuffle the data (will be set False in case of distributed training).

        Returns:
            Prepared dataloader object.
        """
        sampler = torch.utils.data.DistributedSampler(self, shuffle=shuffle) if is_distributed else None
        shuffle = shuffle if sampler is None else False

        dataloader = torch.utils.data.DataLoader(
            self,
            num_workers=1,
            sampler=sampler,
            batch_size=batch_size,
            pin_memory=False,
            drop_last=True,
            shuffle=shuffle
        )

        return dataloader

    def __len__(self):
        return len(self.audio_files)
