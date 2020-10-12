from argparse import ArgumentParser
from pathlib import Path

from torch.nn import CTCLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

from peach.datamodules import LJSpeechDataModule
from peach.models import JasperRecognizer, QuartzNetRecognizer
from peach.trainer import Trainer


def main(args):
    device = args['device']
    learning_rate = args['learning_rate']
    max_epoch = args['max_epoch']

    datamodule = LJSpeechDataModule(
        data_dir=Path("data/LJSpeech-1.1"),
        batch_size=16,
        num_workers=4,
    )
    datamodule.setup()
    model = JasperRecognizer(
        b=10,
        r=5,
        in_channels=None,    #TODO
        out_channels=None,
    ).to(device)
    criterion = CTCLoss    #TODO? CTCLoss(blank=28).to(device)
    optimizer = AdamW(
        params=model.parameters(),
        lr=learning_rate,
    )
    scheduler = StepLR()
    trainer = Trainer(
        max_epoch=max_epoch,
    )

    trainer.fit(
        model=model,
        datamodule=datamodule,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
    )


if __name__ == "__main__":
    args = {
        'learning_rate': 3e-4,
        'max_epoch': 1,
    }
    parser = ArgumentParser() #TODO

    main(args)

