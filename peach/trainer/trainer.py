from pathlib import Path

import neptune
import numpy as np
import torch
from torchaudio.transforms import MelSpectrogram
from tqdm import tqdm


class Trainer:
    def __init__(
            self,
            device,
            logger,
            version,
        ):
        self.device = device
        self.logger = logger
        self.version = version
        self.mel_spectrogramer = MelSpectrogram(
        ).to(self.device)

    def save_checkpoint(
            self,
            model,
            optimizer,
            epoch: int,
            checkpoints_dir: Path,
        ):
        checkpoint = {
            'model': model,
            'optimizer': optimizer,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }
        checkpoint_path = checkpoints_dir / f"v{self.version}-e{epoch}.hdf5"
        torch.save(checkpoint, checkpoint_path)

    def training_epoch(
            self,
            model,
            train_dataloader,
            optimizer,
        ):
        model.train()

        for batch_idx, batch in enumerate(tqdm(train_dataloader)):
            loss = model.training_step(batch, batch_idx)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            model.training_step_end()

        model.training_epoch_end()

    def validation_epoch(
            self,
            model,
            val_dataloader,
        ):
        model.eval()

        for batch_idx, batch in enumerate(tqdm(val_dataloader)):
            loss = model.validation_step(batch, batch_idx)
            model.validation_step_end()

        model.validation_epoch_end()

    def fit(
            self,
            model,
            datamodule,
            criterion,
            optimizer,
            scheduler,
        ):
        train_dataloader = datamodule.train_dataloader()
        val_dataloader = datamodule.val_dataloader()
        optimizer = model.configure_optimizers()

        for epoch in range(1, self.n_epochs + 1):
            self.training_epoch(
                model=model,
                train_dataloader=train_dataloader,
                optimizer=optimizer,
            )
            self.validation_epoch(
                model=model,
                val_dataloader=val_dataloader,
            )
            self.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                checkpoints_dir=Path.cwd() / "models",
            )

        return model

    def predict(
            self,
            model,
            datamodule,
        ):
        pass

