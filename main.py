from argparse import ArgumentParser
from pathlib import Path

import torch

from peach.datamodules import LJSpeechDataModule
from peach.loggers import NeptuneLogger
from peach.models import JasperRecognizer, QuartzNetRecognizer
from peach.trainer import Trainer


def main(args):
    device = args['device']
    learning_rate = args['learning_rate']
    max_epoch = args['max_epoch']
    verbose = args['verbose']
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
    logger = NeptuneLogger(
        api_key=None,
        project_name=None,
    )
    trainer = Trainer(
        logger=logger,
        max_epoch=max_epoch,
        verbose=verbose,
        version=version,
    )

    trainer.fit(
        model=model,
        datamodule=datamodule,
    )


if __name__ == "__main__":
    args = dict(
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        learning_rate=3e-4,
        max_epoch=1,
        verbose=False,
        version='0.1.0',
    )
    parser = ArgumentParser() #TODO

    main(args)

