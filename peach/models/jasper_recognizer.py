from torch.nn import (
    BatchNorm1d,
    Conv1d,
    Dropout,
    Module,
    ReLU,
    Sequential,
)


class JasperSubBlock(Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
        ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sequential = Sequential(
            Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=(3),
                padding=1,
            ),
            BatchNorm1d(num_features=self.out_channels),
            ReLU(inplace=True),
            Dropout(p=0.2, implace=True),
        )

    def forward(self, x):
        output = self.sequential(x)

        return output


class JasperBlock(Module):
    def __init__(
            self,
            r: int,
            in_channels: int,
            out_channels: int,
        ):
        super().__init__()
        self.r = r
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.subblocks_list = list()
        for i in range(self.r):
            self.subblocks_list.append(JasperSubBlock())
        self.subblocks_list.append() #TODO

        self.subblocks = Sequential(*self.subblocks_list)
        self.residual_connection = Sequential(
            Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=(1, 1),
            ),
            BatchNorm1d(), #TODO
        )

    def forward(self, x):#TODO
        residual = x
        x_1 = self.subblocks(x)

        output = self.sequential(x) + residual

        return output


class JasperRecognizer(Module):
    def __init__(
            self,
            b: int=10,
            r: int=5,
        ): #TODO
        self.prolog
        self.sequential
        self.epilog = Sequential(
            
        )

    def forward(self):
        pass

