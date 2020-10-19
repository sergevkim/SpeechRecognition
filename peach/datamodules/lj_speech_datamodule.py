import string
from pathlib import Path
from PIL import Image

import einops
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from peach.utils import TokenConverter


def zero_padding(sequence, new_length):
    padded_sequence = torch.zeros(new_length)
    padded_sequence[:new_length] = sequence[:new_length]

    return padded_sequence



class LJSpeechDataset(Dataset):
    def __init__(
            self,
            filenames,
            targets,
            max_target_length=100,
            max_waveform_length=100000,
        ):
        self.filenames = filenames
        self.targets = targets
        self.max_waveform_length = max_waveform_length

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        waveform, sample_rate = torchaudio.load(filename)
        waveform = einops.rearrange(waveform, 'b x -> (b x)')
        target = TokenConverter.symbols2numbers(
            symbols=self.targets[idx],
        )

        waveform_new_length = min(len(waveform), self.max_waveform_length)
        target_new_length = min(len(target), self.max_target_length)
        waveform_length = Tensor(waveform_new_length)
        target_length = Tensor(target_new_length)

        padded_waveform = zero_padding(
            sequence=waveform,
            new_length=waveform_new_length,
        )
        padded_target = zero_padding(
            sequence=target,
            new_length=target_new_length,
        )

        result = (
            padded_waveform,
            padded_target,
            waveform_length,
            target_length,
        )

        return result


class LJSpeechDataModule:
    def __init__(
            self,
            data_dir: Path,
            batch_size: int,
            num_workers: int,
        ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        wavs_dir = self.data_dir / "wavs"
        targets_path = self.data_dir / "metadata.csv"
        wav_filenames = list(str(p) for p in wavs_dir.glob('*.wav'))
        wav_filenames.sort()
        targets_file = open(targets_path, 'r')

        targets = list()

        for i in range(len(wav_filenames)):
            line = targets_file.readline()
            table = str.maketrans('', '', string.punctuation)
            target = line.split('|')[-1].lower().translate(table)[:-1]
            targets.append(target)

        data = dict(
            filenames=wav_filenames,
            targets=targets,
        )

        return data

    def setup(
            self,
            val_ratio,
        ):
        data = self.prepare_data()
        wav_filenames = data['filenames']
        targets = data['targets']

        full_dataset = LJSpeechDataset(
            filenames=wav_filenames,
            targets=targets,
        )

        full_size = len(full_dataset)
        val_size = int(val_ratio * full_size)
        train_size = full_size - val_size

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset=full_dataset,
            lengths=[train_size, val_size],
        )

    def train_dataloader(self):
        train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        return val_dataloader

    def test_dataloader(self):
        pass

