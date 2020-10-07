from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm


class Trainer:
    def __init__(
            self,
            device,
            version,
            writer=None,
        ):
        self.device = device
        self.version = version
        self.writer = writer

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

    def training_phase(
            self,
            train_dataloader,
            model,
            criterion,
            optimizer,
        ):
        model.train()

        for batch_idx, batch in enumerate(tqdm(train_dataloader)):
            spectrograms, labels = batch
            spectrograms = spectrograms.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            predictions = model(spectrograms)

            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

    def validation_phase(
            self,
            val_dataloader,
            model,
        ):
        model.eval()

        for batch_idx, batch in enumerate(tqdm(val_dataloader)):
            spectrograms, labels = batch
            spectrograms = spectrograms.to(self.device)
            labels = labels.to(self.device)

            predictions = model(spectrograms)
            #TODO

    def fit(
            self,
            datamodule,
            model,
            criterion,
            optimizer,
        ):
        train_dataloader = datamodule.train_dataloader()
        val_dataloader = datamodule.val_dataloader()

        for epoch in range(1, self.n_epochs + 1):
            self.training_phase(
                train_dataloader=train_dataloader,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
            )
            self.validation_phase(
                val_dataloader=val_dataloader,
                model=model,
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
            datamodule,
            model,
        ):
        pass

