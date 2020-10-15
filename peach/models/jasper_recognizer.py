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
            dropout_p: int,
        ):
        super().__init__()
        self.r = r
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_width = kernel_width
        self.dropout_p = dropout_p

        self.subblocks_list = list()
        for i in range(self.r):
            self.subblocks_list.append(JasperSubBlock(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_width=self.kernel_width,
                dropout_p=self.dropout_p,
            ))

        self.subblocks = Sequential(*self.subblocks_list)
        self.first_half_subblock = Sequential(
            Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_width=self.kernel_width,
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
            device=torch.device('cuda'),
        ):
        self.device = device
        self.criterion = CTCLoss().to(delf.device) #TODO
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

        self.b1 = Sequential(
            JasperBlock(
                in_channels=256,
                out_channels=256,
                kernel_width=11,
                dropout_p=0.2,
            ),
            JasperBlock(
                in_channels=256,
                out_channels=256,
                kernel_width=11,
                dropout_p=0.2,
            ),
        )
        self.b2 = Sequential(
            JasperBlock(
                in_channels=256,
                out_channels=384,
                kernel_width=13,
                dropout_p=0.2,
            ),
            JasperBlock(
                in_channels=384,
                out_channels=384,
                kernel_width=13,
                dropout_p=0.2,
            ),
        )
        self.b3 = Sequential(
            JasperBlock(
                in_channels=384,
                out_channels=512,
                kernel_width=17,
                dropout_p=0.2,
            ),
            JasperBlock(
                in_channels=512,
                out_channels=512,
                kernel_width=17,
                dropout_p=0.2,
            ),
        )
        self.b4 = Sequential(
            JasperBlock(
                in_channels=512,
                out_channels=640,
                kernel_width=21,
                dropout_p=0.3,
            ),
            JasperBlock(
                in_channels=640,
                out_channels=640,
                kernel_width=21,
                dropout_p=0.3,
            ),
        )
        self.b5 = Sequential(
            JasperBlock(
                in_channels=640,
                out_channels=768,
                kernel_width=25,
                dropout_p=0.3,
            ),
            JasperBlock(
                in_channels=768,
                out_channels=768,
                kernel_width=25,
                dropout_p=0.3,
            ),
        )

        self.blocks = Sequential(
            self.b1,
            self.b2,
            self.b3,
            self.b4,
            self.b5,
        )
        self.prolog = JasperSubBlock(
            in_channels=self.in_channels,
            out_channels=256,
            kernel_width=11,
            dropout_p=0.2,
            stride=2,
        )
        self.epilog = Sequential(
            JasperSubBlock(
                in_channels=768,
                out_channels=896,
                kernel_width=29,
                dropout_p=0.4,
                dilation=2,
            ),
            JasperSubBlock(
                in_channels=896,
                out_channels=1024,
                kernel_width=1,
                dropout_p=0.4,
            ),
            JasperSubBlock(
                in_channels=1024,
                out_channels=self.out_channels,
                kernel_width=1,
                dropout_p=0,
            ),
        )

    def forward(self, x):
        x_1 = self.prolog(x)
        x_2 = self.blocks(x_1)
        x_3 = self.epilog(x_2)

        return x_3

    def training_step(self, batch, batch_idx):
        waveforms, labels = batch
        waveforms = waveforms.to(device)
        labels = labels.to(device)
        mel_spectrograms = self.mel_spectrogramer(waveforms)

        predictions = self(mel_spectrograms)
        loss = criterion(predictions, labels)

        return loss

    def training_step_end(self):
        pass

    def training_epoch_end(self):
        pass

    def validation_step(self, batch, batch_idx):
        waveforms, labels = batch
        waveforms = waveforms.to(self.device)
        labels = labels.to(self.device)
        mel_spectrograms = self.mel_spectrogramer(waveforms)

        predictions = self(mel_spectrograms)
        loss = criterion(predictions, labels)

        return loss

    def validation_step_end(self):
        pass

    def validation_epoch_end(self):
        pass

    def configure_optimizers(self):
        optimizer = AdamW(
            params=model.parameters(),
            lr=learning_rate,
        )

        return optimizer

