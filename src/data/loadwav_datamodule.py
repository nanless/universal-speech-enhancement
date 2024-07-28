from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from .components.collate import pad_to_longest_monaural_inference
from .components.loadwav_dataset import LoadWavDataset


class LoadWavDataModule(LightningDataModule):
    def __init__(
        self,
        list_path=None,
        json_path=None,
        data_folder=None,
        input_json_list=None,
        input_plain_list=None,
        normalize=False,
        min_duration_seconds=None,
        max_duration_seconds=None,
        sampling_rate=None,
        output_resample=False,
        output_resample_rate=None,
        target_folder=None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_test = LoadWavDataset(
            list_path=list_path,
            json_path=json_path,
            data_folder=data_folder,
            input_json_list=input_json_list,
            input_plain_list=input_plain_list,
            normalize=normalize,
            min_duration_seconds=min_duration_seconds,
            max_duration_seconds=max_duration_seconds,
            sampling_rate=sampling_rate,
            output_resample=output_resample,
            output_resample_rate=output_resample_rate,
            target_folder=target_folder,
        )

        self.batch_size_per_device = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

    def predict_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=pad_to_longest_monaural_inference,
            shuffle=False,
        )
