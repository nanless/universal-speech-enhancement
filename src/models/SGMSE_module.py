import os

import numpy as np
import soundfile as sf
import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MinMetric


class SGMSEModule(LightningModule):
    def __init__(
        self,
        Score: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        super().__init__()

        self.Score = Score
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.compile = compile

    def configure_optimizers(self):
        Score_optimizer = self.optimizer(params=self.Score.parameters())
        Score_scheduler = self.scheduler(optimizer=Score_optimizer)

        return [
            {
                "optimizer": Score_optimizer,
                "lr_scheduler": {
                    "scheduler": Score_scheduler,
                    "monitor": "val/loss_Score_epoch",
                    "interval": "epoch",
                    "frequency": 1,
                },
            },
        ]

    def get_score_loss(self, batch: dict) -> torch.Tensor:
        loss = self.Score.train_step(batch)
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        Score_opt = self.optimizers()

        score_loss = self.get_score_loss(batch)
        self.log("train/loss_Score", score_loss, on_step=True, on_epoch=True, prog_bar=True)
        # log学习率
        lr = Score_opt.param_groups[0]["lr"]
        self.log("lr", lr, on_step=False, on_epoch=True, prog_bar=True)
        return score_loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        score_loss = self.get_score_loss(batch)
        self.log("val/loss_Score", score_loss, on_step=True, on_epoch=True, prog_bar=True)


    def test_step(self, batch: dict, batch_idx: int) -> None:
        score_loss = self.get_score_loss(batch)
        self.log("test/loss_Score", score_loss, on_step=True, on_epoch=True, prog_bar=True)

    @torch.no_grad()
    def predict_step(self, batch: dict, batch_idx: int) -> None:
        torch.cuda.empty_cache()

        batch = self.Score.sample(batch)

        for i, enhanced in enumerate(batch["enhanced"]):
            noisy_path = batch["audio_path"][i]
            sample_length = batch["sample_length"][i]
            sample_rate = batch["sampling_rate"][i]
            enhanced_path = noisy_path.replace(batch["data_folder"], batch["target_folder"])
            if not os.path.exists(os.path.dirname(enhanced_path)):
                os.makedirs(os.path.dirname(enhanced_path))
            enhanced_wav = batch["enhanced"][i].detach().cpu().numpy().astype(np.float32)
            enhanced_wav = enhanced_wav[:sample_length]
            sf.write(enhanced_path, enhanced_wav, sample_rate)

        return batch

    def on_train_start(self) -> None:
        pass

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def on_validation_epoch_end(self) -> None:
        pass

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.compile and stage == "fit":
            self.Score = torch.compile(self.Score)
