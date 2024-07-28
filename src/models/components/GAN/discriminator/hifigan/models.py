import torch
import torch.nn as nn

from .open_models import BandwidthExtender, Discriminator


class Generator(nn.Module):
    def __init__(self, input_sample_rate):
        super().__init__()
        self.bandwidth_extender = BandwidthExtender()
        self.bandwidth_extender.apply_weightnorm()
        self.input_sample_rate = input_sample_rate

    def forward(self, batch_data):
        x = batch_data["perturbed"].unsqueeze(1)
        enhanced = self.bandwidth_extender(x, self.input_sample_rate)
        batch_data["enhanced"] = enhanced.squeeze(1)
        batch_data["output_sampling_rate"] = self.bandwidth_extender.sample_rate
        return batch_data

    def train(self, mode=True):
        # 自定义训练行为
        super().train(mode)
        # if not weight normed before, the weight norm will be applied
        self.bandwidth_extender.train(mode)
        if mode:
            try:
                self.bandwidth_extender.apply_weightnorm()
            except Exception as e:
                print("weight norm already applied", e)

    def eval(self):
        super().eval()
        self.bandwidth_extender.eval()
        try:
            self.bandwidth_extender.remove_weightnorm()
        except Exception as e:
            print("weight norm already removed", e)


class DiscriminatorN(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator = Discriminator()

    def forward_fake(self, batch_data):
        x = batch_data["enhanced"].unsqueeze(1)
        logits, feature_list = self.discriminator(x)
        batch_data["predicted_enhanced_logits"] = logits
        batch_data["predicted_enhanced_feature_list"] = feature_list
        return batch_data

    def forward_real(self, batch_data):
        x = batch_data["clean"].unsqueeze(1)
        logits, feature_list = self.discriminator(x)
        batch_data["predicted_clean_logits"] = logits
        batch_data["predicted_clean_feature_list"] = feature_list
        return batch_data

    def forward(self, batch_data):
        batch_data = self.forward_fake(batch_data)
        batch_data = self.forward_real(batch_data)
        return batch_data
