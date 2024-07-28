import os

import numpy as np
import soundfile as sf
import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MinMetric


class GANModule(LightningModule):
    def __init__(
        self,
        G: torch.nn.Module,
        D: torch.nn.Module,
        G_optimizer: torch.optim.Optimizer,
        D_optimizer: torch.optim.Optimizer,
        G_scheduler: torch.optim.lr_scheduler,
        D_scheduler: torch.optim.lr_scheduler,
        G_criterion: torch.nn.Module,
        D_criterion: torch.nn.Module,
        compile: bool,
        accumulate_grad_batches: int = 1,
        rewrite_lr = False,
        G_lr = None,
        D_lr = None,
    ) -> None:
        super().__init__()

        self.automatic_optimization = False
        self.accumulate_grad_batches = accumulate_grad_batches
        self.batch_count = 0

        self.G = G
        self.D = D
        self.G_optimizer = G_optimizer
        self.G_scheduler = G_scheduler
        self.D_optimizer = D_optimizer
        self.D_scheduler = D_scheduler

        self.G_criterion = G_criterion
        self.D_criterion = D_criterion

        self.compile = compile

        self.rewrite_lr = rewrite_lr

        if self.rewrite_lr:
            self.G_lr = G_lr
            self.D_lr = D_lr

    def load_state_dict(self, state_dict, strict=True):
        model_state_dict = self.state_dict()
        for name, param in state_dict.items():
            if name in model_state_dict:
                if model_state_dict[name].size() == param.size():
                    model_state_dict[name].copy_(param)
                else:
                    print(f"Skipping parameter '{name}' due to size mismatch: {param.size()} vs {model_state_dict[name].size()}")
            else:
                print(f"Skipping parameter '{name}' as it is not found in the current model.")
        super().load_state_dict(model_state_dict, strict=False)

    def configure_optimizers(self):
        G_optimizer = self.G_optimizer(params=self.G.parameters())
        D_optimizer = self.D_optimizer(params=self.D.parameters())

        G_scheduler = self.G_scheduler(optimizer=G_optimizer)
        D_scheduler = self.D_scheduler(optimizer=D_optimizer)

        return [
            {"optimizer": G_optimizer, "lr_scheduler": G_scheduler},
            {"optimizer": D_optimizer, "lr_scheduler": D_scheduler},
        ]

    def get_gen_loss(self, batch: dict) -> torch.Tensor:
        batch = self.D.forward_fake(batch)
        batch = self.D.forward_real(batch)
        batch = self.G_criterion(batch)
        return batch

    def get_disc_loss(self, batch: dict) -> torch.Tensor:
        batch = self.D.forward_fake(batch)
        batch = self.D.forward_real(batch)
        batch = self.D_criterion(batch)
        return batch["loss_D"]

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        G_opt, D_opt = self.optimizers()

        batch = self.G(batch)

        disc_batch = {k: v for (k, v) in batch.items()}
        disc_batch["fake"] = batch["fake"].detach()
        self.toggle_optimizer(D_opt)
        disc_loss = self.get_disc_loss(disc_batch)
        self.log("train/loss_D", disc_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.manual_backward(disc_loss)
        if (self.batch_count + 1) % self.accumulate_grad_batches == 0:
            D_opt.step()
            D_opt.zero_grad()
        self.untoggle_optimizer(D_opt)

        self.toggle_optimizer(G_opt)
        batch = self.get_gen_loss(batch)
        gen_loss = batch["loss_G"]
        for k, v in batch.items():
            if k.startswith("loss_"):
                self.log(f"train/{k}", v, on_step=True, on_epoch=True, prog_bar=True)
        self.manual_backward(gen_loss)
        if (self.batch_count + 1) % self.accumulate_grad_batches == 0:
            G_opt.step()
            G_opt.zero_grad()
            self.batch_count = 0
        else:
            self.batch_count += 1
        self.untoggle_optimizer(G_opt)

        self.log("lr/G", G_opt.param_groups[0]["lr"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("lr/D", D_opt.param_groups[0]["lr"], on_step=True, on_epoch=True, prog_bar=True)

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        batch = self.G(batch)

        batch = self.get_gen_loss(batch)
        gen_loss = batch["loss_G"]
        for k, v in batch.items():
            if k.startswith("loss_"):
                self.log(f"val/{k}", v, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch: dict, batch_idx: int) -> None:
        batch = self.G(batch)

        batch = self.get_gen_loss(batch)
        gen_loss = batch["loss_G"]
        for k, v in batch.items():
            if k.startswith("loss_"):
                self.log(f"test/{k}", v, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

    @torch.no_grad()
    def predict_step(self, batch: dict, batch_idx: int) -> None:
        torch.cuda.empty_cache()

        batch = self.G(batch)
        for i, enhanced in enumerate(batch["fake"]):
            noisy_path = batch["audio_path"][i]
            sample_length = batch["sample_length"][i]
            sample_rate = batch["sampling_rate"][i]
            enhanced_path = noisy_path.replace(batch["data_folder"], batch["target_folder"])
            if not os.path.exists(os.path.dirname(enhanced_path)):
                os.makedirs(os.path.dirname(enhanced_path))
            enhanced_wav = batch["fake"][i].detach().cpu().numpy().astype(np.float32)
            enhanced_wav = enhanced_wav[:sample_length]
            sf.write(enhanced_path, enhanced_wav, sample_rate)

        return batch

    def on_train_start(self) -> None:
        if self.rewrite_lr:
            G_opt, D_opt = self.optimizers()
            for opt in [G_opt, D_opt]:
                for param_group in opt.param_groups:
                    param_group["lr"] = self.G_lr if opt is G_opt else self.D_lr

    def on_train_epock_start(self) -> None:
        torch.cuda.empty_cache()

    def on_train_epoch_end(self) -> None:
        G_scheduler, D_scheduler = self.lr_schedulers()
        G_scheduler.step()
        D_scheduler.step()

    def on_validation_epoch_end(self) -> None:
        pass

    def on_test_epoch_end(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        if self.compile and stage == "fit":
            self.G = torch.compile(self.G)
            self.D = torch.compile(self.D)
