from collections import OrderedDict

import torch
from torch.nn import (
    BatchNorm1d,
    Conv1d,
    CTCLoss,
    Dropout,
    Module,
    ReLU,
    Sequential,
)
from torch.optim import AdamW
from torchaudio.transforms import MelSpectrogram

from peach.utils.metric_calculator import MetricCalculator


class JasperSubBlock(Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_width: int,
            dropout_p: float,
            stride: int=1,
            dilation: int=1,
        ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_width = kernel_width
        self.dropout_p = dropout_p
        self.stride = stride
        self.dilation = dilation

        self.sequential = Sequential(
            Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_width,
                stride=self.stride,
                dilation=self.dilation,
            ),
            BatchNorm1d(num_features=self.out_channels),
            ReLU(inplace=True),
            Dropout(p=self.dropout_p, inplace=True),
        )

    def forward(self, x):
        x_1 = self.sequential(x)

        return x_1


class JasperBlock(Module):
    def __init__(
            self,
            r: int,
            in_channels: int,
            out_channels: int,
            kernel_width: int,
            dropout_p: float=0,
        ):
        super().__init__()
        self.r = r
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_width = kernel_width
        self.dropout_p = dropout_p
        self.subblocks_ordered_dict = OrderedDict()

        for i in range(self.r):
            self.subblocks_ordered_dict[f'subblock_{i}'] = JasperSubBlock(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_width=self.kernel_width,
                dropout_p=self.dropout_p,
            )

        self.subblocks = Sequential(self.subblocks_ordered_dict)
        self.first_half_subblock = Sequential(
            Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_width,
            ),
            BatchNorm1d(num_features=self.out_channels),
        )
        self.second_half_subblock = Sequential(
            ReLU(inplace=True),
            Dropout(p=self.dropout_p, inplace=True),
        )

    def forward(self, x):
        x_1 = self.subblocks(x)
        x_2 = self.residual_connection(x)
        x_3 = self.first_half_subblock(x_1)
        x_4 = x_2 + x_3
        x_5 = self.second_half_subblock(x_4)

        return x_5


class JasperRecognizer(Module):
    def __init__(
            self,
            b: int=10,
            r: int=5,
            in_channels: int=100,
            out_channels: int=200,
            learning_rate: float=3e-4,
            device=torch.device('cpu'),
        ):
        super().__init__()
        self.device = device
        self.learning_rate = learning_rate
        self.criterion = CTCLoss().to(self.device)
        self.mel_spectrogramer = MelSpectrogram(
            n_fft=1024,
            sample_rate=22000,
            win_length=1024,
            hop_length=256,
            f_min=0,
            f_max=800,
            n_mels=80,
        ).to(self.device)

        self.b = b
        self.r = r
        self.in_channels = in_channels
        self.out_channels = out_channels

        in_channels_list = [256, 256, 384, 512, 640]
        out_channels_list = [256, 384, 514, 640, 768]
        kernel_widths_list = [11, 13, 17, 21, 25]
        dropouts_list = [0.2, 0.2, 0.2, 0.3, 0.3]
        self.blocks_ordered_dict = OrderedDict()

        for i in range(5):
            self.blocks_ordered_dict[f'block_{i}_0'] = JasperBlock(
                r=self.r,
                in_channels=in_channels_list[i],
                out_channels=out_channels_list[i],
                kernel_width=kernel_widths_list[i],
                dropout_p=dropouts_list[i],
            )
            self.blocks_ordered_dict[f'block_{i}_1'] = JasperBlock(
                r=self.r,
                in_channels=out_channels_list[i],
                out_channels=out_channels_list[i],
                kernel_width=kernel_widths_list[i],
                dropout_p=dropouts_list[i],
            )

        self.prolog = JasperSubBlock(
            in_channels=self.in_channels,
            out_channels=256,
            kernel_width=11,
            dropout_p=0.2,
            stride=2,
        )
        self.blocks = Sequential(self.blocks_ordered_dict)
        self.epilog = Sequential(OrderedDict(
            subblock_0=JasperSubBlock(
                in_channels=768,
                out_channels=896,
                kernel_width=29,
                dropout_p=0.4,
                dilation=2,
            ),
            subblock_1=JasperSubBlock(
                in_channels=896,
                out_channels=1024,
                kernel_width=1,
                dropout_p=0.4,
            ),
            subblock_2=JasperSubBlock(
                in_channels=1024,
                out_channels=self.out_channels,
                kernel_width=1,
                dropout_p=0,
            ),
        ))

    def forward(self, x):
        x_1 = self.prolog(x)
        x_2 = self.blocks(x_1)
        x_3 = self.epilog(x_2)

        return x_3

    def training_step(self, batch, batch_idx):
        waveforms, targets, waveform_lengths, target_lengths = batch
        waveforms = waveforms.to(device)
        targets = targets.to(device) #TODO lengths.to(device)?
        mel_spectrograms = self.mel_spectrogramer(waveforms)

        predictions = self(mel_spectrograms)
        log_probs = torch.nn.functional.log_softmax(predictions)
        answers = peach.utils.find_best_path(log_probs) #TODO find best path
        loss = criterion(
            log_probs=log_probs,
            targets=targets,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
        )

        cer = MetricCalculator.calculate_cer(
            answers=answers,
            targets=targets,
        )
        wer = MetricCalculator.calculate_wer(
            answers=answers,
            targets=targets,
        )

        return loss, cer, wer

    def training_step_end(self):
        pass

    def training_epoch_end(self):
        print("Training epoch is over!")

    def validation_step(self, batch, batch_idx):
        '''
        waveforms, labels = batch
        waveforms = waveforms.to(self.device)
        labels = labels.to(self.device)
        mel_spectrograms = self.mel_spectrogramer(waveforms)

        predictions = self(mel_spectrograms)
        loss = criterion(predictions, labels)

        return loss
        '''
        pass

    def validation_step_end(self):
        pass

    def validation_epoch_end(self):
        print("Training epoch is over!")

    def configure_optimizers(self):
        optimizer = AdamW(
            params=self.parameters(),
            lr=self.learning_rate,
        )

        return optimizer

