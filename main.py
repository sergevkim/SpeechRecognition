from argparse import ArgumentParser
from pathlib import Path

import neptune
import torch

from peach.datamodules import LJSpeechDataModule
from peach.models import JasperRecognizer, QuartzNetRecognizer
from peach.trainer import Trainer


def main(args):
    device = args['device']
    learning_rate = args['learning_rate']
    max_epoch = args['max_epoch']
    version = args['version']

    model = JasperRecognizer(
        b=10,
        r=5,
        device=device,
    ).to(device)
    datamodule = LJSpeechDataModule(
        data_dir=Path("data/LJSpeech-1.1"),
        batch_size=16,
        num_workers=4,
    )
    datamodule.setup(val_ratio=0.1)
    logger = None #TODO
    trainer = Trainer(
        logger=logger,
        max_epoch=max_epoch,
        version=version,
    )

    trainer.fit(
        model=model,
        datamodule=datamodule,
    )


if __name__ == "__main__":
    args = dict(
        device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
        learning_rate=3e-4,
        max_epoch=1,
        version='0.1.0',
    )
    parser = ArgumentParser() #TODO

    main(args)

