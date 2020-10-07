from torch.nn import CTCLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

from peach.datamodules import SimpleDataModule
from peach.models import SimpleRecognizer, QuartzNetRecognizer
from peach.trainer import Trainer


def main():
    datamodule = SimpleDataModule()
    model = SimpleRecognizer(
        in_channels=None,    #TODO
        out_channels=None,
    )
    criterion = CTCLoss()
    optimizer = AdamW(
        params=model.parameters(),
        lr=args['learning_rate'],
    )
    scheduler = StepLR()
    trainer = Trainer(
        max_epoch = args['max_epoch'],
    )

    trainer.fit(
        datamodule=datamodule,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
    )


if __name__ == "__main__":
    args = {
        'learning_rate': 3e-4,
        'max_epoch': 1,
    }
    main(args)

