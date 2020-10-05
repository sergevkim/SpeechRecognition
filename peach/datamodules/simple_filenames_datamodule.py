from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


class SimpleFilenamesDataset(Dataset):
    def __init__(
            self,
            filenames,
            labels=[],
            transform_list=[
                ToTensor(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ],
        ):
        self.filenames = filenames
        self.labels = labels
        self.transform = Compose(transform_list)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_filename = self.filenames[idx]
        img = Image.open(img_filename)
        img = self.transform(img)

        if self.labels:
            label = self.labels[idx]
            return (img, label)
        else:
            return (img, img_filename)

