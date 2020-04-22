import torch

from torch import Tensor

from tacotron2 import hparams as taco_hparams
from waveglow.models import _layers as layers


class WaveGlow(torch.nn.Module):
    def __init__(self, hparams: taco_hparams.HParams):
        super(WaveGlow, self).__init__()

        self.upsample = torch.nn.ConvTranspose1d(hparams.n_mel_channels, hparams.n_mel_channels, hparams.win_length,
                                                 stride=hparams.hop_length)
        assert (hparams.n_group % 2 == 0)
        self.n_flows = hparams.n_flows
        self.n_group = hparams.n_group
        self.n_early_every = hparams.n_early_every
        self.n_early_size = hparams.n_early_size
        self.WN = torch.nn.ModuleList()
        self.convinv = torch.nn.ModuleList()

        n_half = int(hparams.n_group / 2)

        # Set up layers with the right sizes based on how many dimensions
        # have been output already
        n_remaining_channels = hparams.n_group
        for k in range(hparams.n_flows):
            if k % self.n_early_every == 0 and k > 0:
                n_half = n_half - int(self.n_early_size / 2)
                n_remaining_channels = n_remaining_channels - self.n_early_size
            self.convinv.append(layers.Invertible1x1Conv(n_remaining_channels))
            self.WN.append(layers.WN(n_half, hparams.n_mel_channels * hparams.n_group, **hparams.WN_config))

        self.convinv_inversed = self.convinv[::-1]
        self.WN_inversed = self.WN[::-1]

        self.n_remaining_channels = n_remaining_channels  # Useful during inference
        self.criterion = layers.WaveGlowLoss(hparams.sigma)

    @torch.jit.ignore
    def forward(self, forward_input):
        """
        forward_input[0] = mel_spectrogram:  batch x n_mel_channels x frames
        forward_input[1] = audio: batch x time
        """
        spect, audio = forward_input['x'], forward_input['y']

        #  Upsample spectrogram to size of audio
        spect = self.upsample(spect)
        assert (spect.size(2) >= audio.size(1))
        if spect.size(2) > audio.size(1):
            spect = spect[:, :, :audio.size(1)]

        spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1).permute(0, 2, 1)

        audio = audio.unfold(1, self.n_group, self.n_group).permute(0, 2, 1)
        output_audio = []
        log_s_list = []
        log_det_W_list = []

        for k in range(self.n_flows):
            if k % self.n_early_every == 0 and k > 0:
                output_audio.append(audio[:, :self.n_early_size, :])
                audio = audio[:, self.n_early_size:, :]

            audio, log_det_W = self.convinv[k](audio)
            log_det_W_list.append(log_det_W)

            n_half = int(audio.size(1) / 2)
            audio_0 = audio[:, :n_half, :]
            audio_1 = audio[:, n_half:, :]

            output = self.WN[k](audio_0, spect)
            log_s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = torch.exp(log_s) * audio_1 + b
            log_s_list.append(log_s)

            audio = torch.cat([audio_0, audio_1], 1)

        output_audio.append(audio)

        model_output = torch.cat(output_audio, 1), log_s_list, log_det_W_list

        loss = self.criterion(torch.cat(output_audio, 1), log_s_list, log_det_W_list)

        return model_output, loss

    @torch.jit.export
    def infer(self, spect: Tensor, sigma):

        sigma = 1

        spect = self.upsample(spect)
        # trim conv artifacts. maybe pad spec to kernel multiple
        time_cutoff = self.upsample.kernel_size[0] - self.upsample.stride[0]
        spect = spect[:, :, :-time_cutoff]

        spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1).permute(0, 2, 1)

        audio: Tensor = torch.normal(
            mean=torch.zeros(spect.size(0), self.n_remaining_channels, spect.size(2)),
            std=torch.ones(spect.size(0), self.n_remaining_channels, spect.size(2))
        ).to(spect.device)

        audio = sigma * audio

        for k, (wn, conv) in enumerate(zip(self.WN_inversed, self.convinv_inversed)):

            k = self.n_flows - 1 - k
            audio_size = audio.size(1)

            n_half = int(audio_size / 2)
            audio_0 = audio[:, :n_half, :]
            audio_1 = audio[:, n_half:, :]

            output = wn(audio_0, spect)

            s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = (audio_1 - b) / torch.exp(s)
            audio = torch.cat([audio_0, audio_1], dim=1)

            audio = conv.infer(audio)

            if k % self.n_early_every == 0 and k > 0:
                z = torch.normal(
                    mean=torch.zeros(spect.size(0), self.n_early_size, spect.size(2)),
                    std=torch.ones(spect.size(0), self.n_early_size, spect.size(2))
                ).to(spect.device)
                sigma_: Tensor = sigma * z
                audio = torch.cat([sigma_, audio], dim=1)

        audio = audio.permute(0, 2, 1).contiguous().view(audio.size(0), -1).data
        return audio

    @staticmethod
    def remove_weightnorm(model):
        waveglow = model
        for WN in waveglow.WN:
            WN.start = torch.nn.utils.remove_weight_norm(WN.start)
            WN.in_layers = layers.remove(WN.in_layers)
            WN.cond_layer = torch.nn.utils.remove_weight_norm(WN.cond_layer)
            WN.res_skip_layers = layers.remove(WN.res_skip_layers)
        return waveglow


class WaveGlow(torch.nn.Module):
    def __init__(self, hparams: taco_hparams.HParams):
        super(WaveGlow, self).__init__()

        self.upsample = torch.nn.ConvTranspose1d(hparams.n_mel_channels, hparams.n_mel_channels, hparams.win_length,
                                                 stride=hparams.hop_length)
        assert (hparams.n_group % 2 == 0)
        self.n_flows = hparams.n_flows
        self.n_group = hparams.n_group
        self.n_early_every = hparams.n_early_every
        self.n_early_size = hparams.n_early_size
        self.WN = torch.nn.ModuleList()
        self.convinv = torch.nn.ModuleList()

        n_half = int(hparams.n_group / 2)

        # Set up layers with the right sizes based on how many dimensions
        # have been output already
        n_remaining_channels = hparams.n_group
        for k in range(hparams.n_flows):
            if k % self.n_early_every == 0 and k > 0:
                n_half = n_half - int(self.n_early_size / 2)
                n_remaining_channels = n_remaining_channels - self.n_early_size
            self.convinv.append(layers.Invertible1x1Conv(n_remaining_channels))
            self.WN.append(layers.WN(n_half, hparams.n_mel_channels * hparams.n_group, **hparams.WN_config))

        self.convinv_inversed = self.convinv[::-1]
        self.WN_inversed = self.WN[::-1]

        self.n_remaining_channels = n_remaining_channels  # Useful during inference
        self.criterion = layers.WaveGlowLoss(hparams.sigma)

    @torch.jit.ignore
    def forward(self, forward_input):
        """
        forward_input[0] = mel_spectrogram:  batch x n_mel_channels x frames
        forward_input[1] = audio: batch x time
        """
        spect, audio = forward_input['x'], forward_input['y']

        #  Upsample spectrogram to size of audio
        spect = self.upsample(spect)
        assert (spect.size(2) >= audio.size(1))
        if spect.size(2) > audio.size(1):
            spect = spect[:, :, :audio.size(1)]

        spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1).permute(0, 2, 1)

        audio = audio.unfold(1, self.n_group, self.n_group).permute(0, 2, 1)
        output_audio = []
        log_s_list = []
        log_det_W_list = []

        for k in range(self.n_flows):
            if k % self.n_early_every == 0 and k > 0:
                output_audio.append(audio[:, :self.n_early_size, :])
                audio = audio[:, self.n_early_size:, :]

            audio, log_det_W = self.convinv[k](audio)
            log_det_W_list.append(log_det_W)

            n_half = int(audio.size(1) / 2)
            audio_0 = audio[:, :n_half, :]
            audio_1 = audio[:, n_half:, :]

            output = self.WN[k](audio_0, spect)
            log_s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = torch.exp(log_s) * audio_1 + b
            log_s_list.append(log_s)

            audio = torch.cat([audio_0, audio_1], 1)

        output_audio.append(audio)

        model_output = torch.cat(output_audio, 1), log_s_list, log_det_W_list

        loss = self.criterion(torch.cat(output_audio, 1), log_s_list, log_det_W_list)

        return model_output, loss

    @torch.jit.export
    def infer(self, spect: Tensor, sigma):

        sigma = 1

        spect = self.upsample(spect)
        # trim conv artifacts. maybe pad spec to kernel multiple
        time_cutoff = self.upsample.kernel_size[0] - self.upsample.stride[0]
        spect = spect[:, :, :-time_cutoff]

        spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1).permute(0, 2, 1)

        audio: Tensor = torch.normal(
            mean=torch.zeros(spect.size(0), self.n_remaining_channels, spect.size(2)),
            std=torch.ones(spect.size(0), self.n_remaining_channels, spect.size(2))
        ).to(spect.device)

        audio = sigma * audio

        for k, (wn, conv) in enumerate(zip(self.WN_inversed, self.convinv_inversed)):

            k = self.n_flows - 1 - k
            audio_size = audio.size(1)

            n_half = int(audio_size / 2)
            audio_0 = audio[:, :n_half, :]
            audio_1 = audio[:, n_half:, :]

            output = wn(audio_0, spect)

            s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = (audio_1 - b) / torch.exp(s)
            audio = torch.cat([audio_0, audio_1], dim=1)

            audio = conv.infer(audio)

            if k % self.n_early_every == 0 and k > 0:
                z = torch.normal(
                    mean=torch.zeros(spect.size(0), self.n_early_size, spect.size(2)),
                    std=torch.ones(spect.size(0), self.n_early_size, spect.size(2))
                ).to(spect.device)
                sigma_: Tensor = sigma * z
                audio = torch.cat([sigma_, audio], dim=1)

        audio = audio.permute(0, 2, 1).contiguous().view(audio.size(0), -1).data
        return audio

    @staticmethod
    def remove_weightnorm(model):
        waveglow = model
        for WN in waveglow.WN:
            WN.start = torch.nn.utils.remove_weight_norm(WN.start)
            WN.in_layers = layers.remove(WN.in_layers)
            WN.cond_layer = torch.nn.utils.remove_weight_norm(WN.cond_layer)
            WN.res_skip_layers = layers.remove(WN.res_skip_layers)
        return waveglow


if __name__ == '__main__':
    hparams = taco_hparams.HParams.from_yaml('../../configs/hparams.default.yaml')
    model = WaveGlow(hparams)
    model_scripted = torch.jit.script(model)

    dummy_input = torch.randn(size=(1, 80, 35))
    dummy_output = model_scripted.infer(dummy_input, sigma=torch.tensor(1))

    print()
