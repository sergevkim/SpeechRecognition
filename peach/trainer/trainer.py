from pathlib import Path
import numpy as np
import pandas as pd
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

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        imgs = imgs.to(self.device)
        labels = labels.to(self.device)

        optimizer.zero_grad()
        predictions = model(imgs)

        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        output = {
            'loss': loss,
        }
        return output

    def fit(
            self,
            criterion,
            train_dataloader,
            val_dataloader,
            model,
            optimizer,
            n_epochs=10,
        ):

        for epoch in range(1, n_epochs + 1):
            running_loss = 0

            model.train()
            for batch_idx, batch in enumerate(tqdm(train_dataloader)):
                imgs, labels = batch
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                predictions = model(imgs)

                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            total = 0
            correct = 0

            model.eval()
            for batch_idx, batch in enumerate(tqdm(val_dataloader)):
                imgs, labels = batch
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                with torch.no_grad():
                    predictions = model(imgs)
                    _, answers = torch.max(input=predictions, dim=1)

                    total += labels.size(0)
                    correct += (answers == labels).sum().item()

                self.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    checkpoints_dir=Path.cwd() / "models",
                )

            print(f"{100 * correct / total}% accuracy on {epoch} epoch")

        return model


    def predict(
            self,
            model,
            test_dataloader,
            labels_test_filename="labels_test.csv",
        ):

        submission = []

        with torch.no_grad():
            model.eval()
            for batch_idx, batch in enumerate(tqdm(test_dataloader)):
                imgs, filenames = batch
                imgs = imgs.to(self.device)
                predictions = model(imgs)
                _, answers = torch.max(input=predictions, dim=1)

                filenames = list(map(lambda p: p.split('/')[-1], filenames))
                answers = list(map(lambda p: str(p).rjust(4, '0'), answers.tolist()))
                submission += list(zip(filenames, answers))

        submission.sort()
        submission = np.array(submission)
        df = pd.DataFrame(data={'Id': submission[:,0], 'Category': submission[:,1]})
        df.to_csv(labels_test_filename, sep=',', index=False)

