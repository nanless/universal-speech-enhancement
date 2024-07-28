import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ....feature.stft import STFTFeature

EPS = 1e-6


class cLN(nn.Module):
    def __init__(self, dimension, eps=1e-8, trainable=True):
        super().__init__()

        self.eps = eps
        if trainable:
            self.gain = nn.Parameter(torch.ones(1, dimension, 1))
            self.bias = nn.Parameter(torch.zeros(1, dimension, 1))
        else:
            self.gain = Variable(torch.ones(1, dimension, 1), requires_grad=False)
            self.bias = Variable(torch.zeros(1, dimension, 1), requires_grad=False)

    def forward(self, input):
        # input size: (Batch, Freq, Time)
        # cumulative mean for each time step

        batch_size = input.size(0)
        channel = input.size(1)
        time_step = input.size(2)

        step_sum = input.sum(1)  # B, T
        step_pow_sum = input.pow(2).sum(1)  # B, T
        cum_sum = torch.cumsum(step_sum, dim=1)  # B, T
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=1)  # B, T

        entry_cnt = np.arange(channel, channel * (time_step + 1), channel)
        entry_cnt = torch.from_numpy(entry_cnt).type(input.type())
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)

        cum_mean = cum_sum / entry_cnt  # B, T
        cum_var = (cum_pow_sum - 2 * cum_mean * cum_sum) / entry_cnt + cum_mean.pow(2)  # B, T
        cum_std = (cum_var + self.eps).sqrt()  # B, T

        cum_mean = cum_mean.unsqueeze(1)
        cum_std = cum_std.unsqueeze(1)

        x = (input - cum_mean.expand_as(input)) / cum_std.expand_as(input)
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""

    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


class MultiRNN(nn.Module):
    """Container module for multiple stacked RNN layers.

    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state. The corresponding output should
                    have shape (batch, seq_len, hidden_size).
        num_layers: int, number of stacked RNN layers. Default is 1.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(
        self, rnn_type, input_size, hidden_size, dropout=0, num_layers=1, bidirectional=False
    ):
        super().__init__()

        self.rnn = getattr(nn, rnn_type)(
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_direction = int(bidirectional) + 1

    def forward(self, input):
        hidden = self.init_hidden(input.size(0))
        self.rnn.flatten_parameters()
        return self.rnn(input, hidden)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if self.rnn_type == "LSTM":
            return (
                Variable(
                    weight.new(
                        self.num_layers * self.num_direction, batch_size, self.hidden_size
                    ).zero_()
                ),
                Variable(
                    weight.new(
                        self.num_layers * self.num_direction, batch_size, self.hidden_size
                    ).zero_()
                ),
            )
        else:
            return Variable(
                weight.new(
                    self.num_layers * self.num_direction, batch_size, self.hidden_size
                ).zero_()
            )


class FCLayer(nn.Module):
    """Container module for a fully-connected layer.

    args:
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, input_size).
        hidden_size: int, dimension of the output. The corresponding output should
                    have shape (batch, hidden_size).
        nonlinearity: string, the nonlinearity applied to the transformation. Default is None.
    """

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity=None):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.FC = nn.Linear(self.input_size, self.hidden_size, bias=bias)
        if nonlinearity:
            self.nonlinearity = getattr(F, nonlinearity)
        else:
            self.nonlinearity = None

        self.init_hidden()

    def forward(self, input):
        if self.nonlinearity is not None:
            return self.nonlinearity(self.FC(input))
        else:
            return self.FC(input)

    def init_hidden(self):
        initrange = 1.0 / np.sqrt(self.input_size * self.hidden_size)
        self.FC.weight.data.uniform_(-initrange, initrange)
        if self.bias:
            self.FC.bias.data.fill_(0)


class DepthConv1d(nn.Module):
    def __init__(
        self, input_channel, hidden_channel, kernel, padding, dilation=1, skip=True, causal=False
    ):
        super().__init__()

        self.causal = causal
        self.skip = skip

        self.conv1d = nn.Conv1d(input_channel, hidden_channel, 1)
        if self.causal:
            self.padding = (kernel - 1) * dilation
        else:
            self.padding = padding
        self.dconv1d = nn.Conv1d(
            hidden_channel,
            hidden_channel,
            kernel,
            dilation=dilation,
            groups=hidden_channel,
            padding=self.padding,
        )
        self.res_out = nn.Conv1d(hidden_channel, input_channel, 1)
        self.nonlinearity1 = nn.PReLU()
        self.nonlinearity2 = nn.PReLU()
        if self.causal:
            self.reg1 = cLN(hidden_channel, eps=1e-08)
            self.reg2 = cLN(hidden_channel, eps=1e-08)
        else:
            self.reg1 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
            self.reg2 = nn.GroupNorm(1, hidden_channel, eps=1e-08)

        if self.skip:
            self.skip_out = nn.Conv1d(hidden_channel, input_channel, 1)

    def forward(self, input):
        output = self.reg1(self.nonlinearity1(self.conv1d(input)))
        if self.causal:
            output = self.reg2(self.nonlinearity2(self.dconv1d(output)[:, :, : -self.padding]))
        else:
            output = self.reg2(self.nonlinearity2(self.dconv1d(output)))
        residual = self.res_out(output)
        if self.skip:
            skip = self.skip_out(output)
            return residual, skip
        else:
            return residual


class TCN(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        BN_dim,
        hidden_dim,
        layer,
        stack,
        kernel=3,
        skip=True,
        causal=False,
        dilated=True,
    ):
        super().__init__()

        # input is a sequence of features of shape (B, N, L)

        # normalization
        if not causal:
            self.LN = nn.GroupNorm(1, input_dim, eps=1e-8)
        else:
            self.LN = cLN(input_dim, eps=1e-8)

        self.BN = nn.Conv1d(input_dim, BN_dim, 1)

        # TCN for feature extraction
        self.receptive_field = 0
        self.dilated = dilated

        self.TCN = nn.ModuleList([])
        for s in range(stack):
            for i in range(layer):
                if self.dilated:
                    self.TCN.append(
                        DepthConv1d(
                            BN_dim,
                            hidden_dim,
                            kernel,
                            dilation=2**i,
                            padding=2**i,
                            skip=skip,
                            causal=causal,
                        )
                    )
                else:
                    self.TCN.append(
                        DepthConv1d(
                            BN_dim,
                            hidden_dim,
                            kernel,
                            dilation=1,
                            padding=1,
                            skip=skip,
                            causal=causal,
                        )
                    )
                if i == 0 and s == 0:
                    self.receptive_field += kernel
                else:
                    if self.dilated:
                        self.receptive_field += (kernel - 1) * 2**i
                    else:
                        self.receptive_field += kernel - 1

        # print("Receptive field: {:3d} frames.".format(self.receptive_field))

        # output layer

        self.output = nn.Sequential(nn.PReLU(), nn.Conv1d(BN_dim, output_dim, 1))

        self.skip = skip

    def forward(self, input):
        # input shape: (B, N, L)

        # normalization
        output = self.BN(self.LN(input))

        # pass to TCN
        if self.skip:
            skip_connection = 0.0
            for i in range(len(self.TCN)):
                residual, skip = self.TCN[i](output)
                output = output + residual
                skip_connection = skip_connection + skip
        else:
            for i in range(len(self.TCN)):
                residual = self.TCN[i](output)
                output = output + residual

        # output layer
        if self.skip:
            output = self.output(skip_connection)
        else:
            output = self.output(output)

        return output


class _LayerNorm(nn.Module):
    """Layer Normalization base class."""

    def __init__(self, channel_size):
        super().__init__()
        self.channel_size = channel_size
        self.gamma = nn.Parameter(torch.ones(channel_size), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(channel_size), requires_grad=True)

    def apply_gain_and_bias(self, normed_x):
        """Assumes input of size `[batch, channel, *]`."""
        return (self.gamma * normed_x.transpose(1, -1) + self.beta).transpose(1, -1)


class CumLN(_LayerNorm):
    """Cumulative Global layer normalization(cumLN)."""

    def forward(self, x):
        """

        Args:
            x (`torch.Tensor`): Shape `[batch, channels, length]`
        Returns:
             `torch.Tensor`: cumLN_x `[batch, channels, length]`
        """
        batch, chan, spec_len = x.size()
        cum_sum = torch.cumsum(x.sum(1, keepdim=True), dim=-1)
        cum_pow_sum = torch.cumsum(x.pow(2).sum(1, keepdim=True), dim=-1)
        cnt = torch.arange(
            start=chan, end=chan * (spec_len + 1), step=chan, dtype=x.dtype, device=x.device
        ).view(1, 1, -1)
        cum_mean = cum_sum / cnt
        cum_var = cum_pow_sum / cnt - cum_mean.pow(2)
        return self.apply_gain_and_bias((x - cum_mean) / (cum_var + EPS).sqrt())


class CumLN2d(_LayerNorm):
    """Cumulative Global layer normalization(cumLN)."""

    def forward(self, x):
        """

        Args:
            x (`torch.Tensor`): Shape `[batch, channels, length, freq]`
        Returns:
             `torch.Tensor`: cumLN_x `[batch, channels, length, freq]`
        """
        batch, chan, spec_len, freq = x.size()
        cum_sum = torch.cumsum(x.sum(1, keepdim=True), dim=2)
        cum_pow_sum = torch.cumsum(x.pow(2).sum(1, keepdim=True), dim=2)
        cnt = torch.arange(
            start=chan, end=chan * (spec_len + 1), step=chan, dtype=x.dtype, device=x.device
        ).view(1, 1, -1, 1)
        cum_mean = cum_sum / cnt
        cum_var = cum_pow_sum / cnt - cum_mean.pow(2)
        return self.apply_gain_and_bias((x - cum_mean) / (cum_var + EPS).sqrt())


class CausalConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=(1, 1),
        dilation=(1, 1),
        groups=1,
        bias=True,
    ):
        super().__init__()
        assert isinstance(kernel_size, tuple)
        assert isinstance(stride, tuple)
        assert isinstance(dilation, tuple)
        self.padding = (kernel_size[0] - 1) * dilation[0], (kernel_size[1] - 1) * dilation[1] // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias
        )

    def forward(self, x):
        if self.padding[0] > 0 or self.padding[1] > 0:
            x = F.pad(x, (self.padding[1], self.padding[1], self.padding[0], 0))
        x = self.conv(x)
        return x


class PixelShuffle(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        batch_size, channels, t, f = x.size()
        new_channels = channels // self.scale_factor
        new_f = f * self.scale_factor

        # 重新排列张量以实现 PixelShuffle
        x = x.view(batch_size, new_channels, self.scale_factor, t, f)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        x = x.view(batch_size, new_channels, t, new_f)
        return x


class PixelShuffleBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        upscale_factor=2,
        kernel_size=(3, 3),
        stride=(1, 1),
    ):
        super().__init__()
        # print(kernel)
        self.conv = CausalConv2d(
            in_channels=in_channels,
            out_channels=out_channels * upscale_factor,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )
        self.ps = PixelShuffle(upscale_factor)

    def forward(self, x):
        # print("in",x.shape)
        x = self.conv(x)
        x = self.ps(x)
        return x


class Projection(nn.Module):
    def __init__(self, in_channles, out_channels, kenal_size=(3, 3)):
        super().__init__()
        self.conv = CausalConv2d(in_channles, out_channels, kenal_size)

    def forward(self, x):
        x = self.conv(x)
        return x


class Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = self.conv(x)
        outputs, gate = x.chunk(2, dim=1)
        return outputs * self.sigmoid(gate)


class SeChannelModule(nn.Module):
    def __init__(self, channels, kernel_size=(3, 1)):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.conv = CausalConv2d(channels, channels, kernel_size=kernel_size, bias=False)
        self.channels = channels

    def forward(self, x):
        # input: B, C, T, F
        inp = x
        x = x.view(x.shape[0], -1, x.shape[-1])
        x = self.pool(x)
        x = x.view(x.shape[0], self.channels, -1, 1)
        return inp * self.conv(x)


class SeFreqModule(nn.Module):
    def __init__(self, channels, kernel_size=(3, 1)):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.conv = CausalConv2d(channels, channels, kernel_size=kernel_size, bias=False)
        self.channels = channels

    def forward(self, x):
        # input: B, C, T, F
        x = x.transpose(1, 3).contiguous()
        inp = x
        x = x.view(x.shape[0], -1, x.shape[-1])
        x = self.pool(x)
        x = x.view(x.shape[0], self.channels, -1, 1)
        x = inp * self.conv(x)
        return x.transpose(1, 3).contiguous()


def get_norm(norm):
    if norm == "BN":
        return nn.BatchNorm2d
    elif norm == "SyncBN":
        return nn.SyncBatchNorm
    elif norm == "CLN":
        return CumLN2d
    elif norm == "IN":
        return nn.InstanceNorm2d
    else:
        raise NotImplementedError(f"Unsupported normalizaiton: {norm}")


class GLFB(nn.Module):
    def __init__(self, channels, kernel_size, stride, dilation, norm, freq_dim):
        super().__init__()

        self.first_block = nn.Sequential(
            get_norm(norm)(channels),
            nn.Conv2d(channels, 2 * channels, kernel_size=1, bias=False),
            CausalConv2d(
                2 * channels,
                2 * channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                groups=2 * channels,
                bias=True,
            ),
            Gate(),
            SeChannelModule(channels, kernel_size=(3, 1)),
            SeFreqModule(freq_dim, kernel_size=(1, 1)),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )

        self.second_block = nn.Sequential(
            get_norm(norm)(channels),
            nn.Conv2d(channels, 2 * channels, kernel_size=1, bias=False),
            Gate(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )

        self.beta = nn.Parameter(torch.ones((1, channels, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.ones((1, channels, 1, 1)), requires_grad=True)

    def forward(self, x):
        x = x + self.first_block(x) * self.beta
        x = x + self.second_block(x) * self.gamma
        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, depth, norm, freq_dim):
        super().__init__()
        self.glfb = nn.Sequential(
            *[
                GLFB(
                    channels=in_channels,
                    kernel_size=kernel_size,
                    stride=(1, 1),
                    dilation=(2**i, 1),
                    norm=norm,
                    freq_dim=freq_dim,
                )
                for i in range(depth)
            ]
        )

        self.conv = nn.Conv2d(
            in_channels, out_channels, (1, 6), stride=(1, 2), padding=(0, 2), bias=False, groups=1
        )

    def forward(self, x):
        x = self.glfb(x)
        skip = x
        x = self.conv(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, depth, norm, freq_dim):
        super().__init__()

        self.deconv = PixelShuffleBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            upscale_factor=2,
            kernel_size=(3, 3),
            stride=(1, 1),
        )

        self.glfb = nn.Sequential(
            *[
                GLFB(
                    channels=out_channels,
                    kernel_size=kernel_size,
                    stride=(1, 1),
                    dilation=(2**i, 1),
                    norm=norm,
                    freq_dim=freq_dim,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x, skip):
        x = self.deconv(x)
        x = x + skip
        x = self.glfb(x)
        return x


class CSMGAN(nn.Module):
    def __init__(
        self,
        in_proj_channels,
        encoder_channels,
        encoder_depths,
        encoder_GLFB_kernel_size,
        TCN_input_dim,
        TCN_BN_dim,
        TCN_hidden_dim,
        TCN_layers,
        TCN_stacks,
        TCN_kernel_size,
        decoder_depths,
        decoder_GLFB_kernel_size,
        GLFB_norm,
        input_freq,
    ):
        super().__init__()

        self.in_proj = CausalConv2d(
            in_channels=2, out_channels=in_proj_channels, kernel_size=(3, 3)
        )

        self.encoder = nn.ModuleList()
        for i, d in enumerate(encoder_depths):
            self.encoder.append(
                DownBlock(
                    encoder_channels[i],
                    encoder_channels[i + 1],
                    kernel_size=encoder_GLFB_kernel_size,
                    depth=d,
                    norm=GLFB_norm,
                    freq_dim=input_freq // 2**i,
                )
            )

        self.bottleneck = TCN(
            input_dim=TCN_input_dim,
            output_dim=TCN_input_dim,
            BN_dim=TCN_BN_dim,
            hidden_dim=TCN_hidden_dim,
            layer=TCN_layers,
            stack=TCN_stacks,
            kernel=TCN_kernel_size,
            skip=True,
            causal=True,
            dilated=True,
        )

        self.decoder = nn.ModuleList()
        for i, h in enumerate(decoder_depths):
            self.decoder.append(
                UpBlock(
                    encoder_channels[-i - 1],
                    encoder_channels[-i - 2],
                    kernel_size=decoder_GLFB_kernel_size,
                    depth=h,
                    norm=GLFB_norm,
                    freq_dim=input_freq // 2 ** (len(encoder_depths) - i - 1),
                )
            )

        self.out_proj = CausalConv2d(
            in_channels=in_proj_channels, out_channels=2, kernel_size=(3, 3)
        )

    def forward(self, x):
        # inputs: B, 2, T, F
        x = self.in_proj(x)
        skips = []
        for encoder in self.encoder:
            x, skip = encoder(x)
            skips.append(skip)

        nbatch, nchannel, ntime, nfreq = x.shape
        x = x.permute(0, 1, 3, 2).contiguous().view(nbatch, nchannel * nfreq, ntime)
        x = self.bottleneck(x)
        x = x.view(nbatch, nchannel, nfreq, ntime).permute(0, 1, 3, 2).contiguous()

        for i, d in enumerate(self.decoder):
            x = d(x, skips[-i - 1])

        x = self.out_proj(x)
        return x


class CSMGAN_Wrapper(nn.Module):
    def __init__(
        self,
        in_proj_channels=8,
        encoder_channels=[8, 8, 16, 16, 24],
        encoder_depths=[1, 2, 1, 2],
        encoder_GLFB_kernel_size=[3, 3],
        TCN_input_dim=720,
        TCN_BN_dim=600,
        TCN_hidden_dim=600,
        TCN_layers=6,
        TCN_stacks=2,
        TCN_kernel_size=3,
        decoder_depths=[1, 2, 1, 2],
        decoder_GLFB_kernel_size=[3, 3],
        GLFB_norm="CLN",
        input_freq=480,
        n_fft=512,
        win_length=512,
        hop_length=128,
        window="hann",
        use_mag_phase=False,
        need_inverse=True,
        freq_high=None,
        sampling_rate=16000,
        compression=None,
        split_subbands=None,
        inverse_keys=["fake"],
    ):
        super().__init__()
        encoder_GLFB_kernel_size = tuple(encoder_GLFB_kernel_size)
        decoder_GLFB_kernel_size = tuple(decoder_GLFB_kernel_size)
        self.feature = STFTFeature(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window=window,
            use_mag_phase=use_mag_phase,
            need_inverse=need_inverse,
            freq_high=freq_high,
            sampling_rate=sampling_rate,
            compression=compression,
            split_subbands=split_subbands,
            inverse_keys=inverse_keys,
        )
        self.net = CSMGAN(
            in_proj_channels=in_proj_channels,
            encoder_channels=encoder_channels,
            encoder_depths=encoder_depths,
            encoder_GLFB_kernel_size=encoder_GLFB_kernel_size,
            TCN_input_dim=TCN_input_dim,
            TCN_BN_dim=TCN_BN_dim,
            TCN_hidden_dim=TCN_hidden_dim,
            TCN_layers=TCN_layers,
            TCN_stacks=TCN_stacks,
            TCN_kernel_size=TCN_kernel_size,
            decoder_depths=decoder_depths,
            decoder_GLFB_kernel_size=decoder_GLFB_kernel_size,
            GLFB_norm=GLFB_norm,
            input_freq=input_freq,
        )

    def forward(self, batch_data):
        """B, t -> B, F, T, 2 -> B, 2, T, F."""
        batch_data = self.feature(batch_data)
        x = batch_data["perturbed_spectra"].permute(0, 3, 2, 1).contiguous()[..., :-1]
        x = self.net(x)
        x = F.pad(x, (0, 1)).permute(0, 3, 2, 1).contiguous()
        batch_data["fake_spectra"] = x
        batch_data = self.feature.inverse(batch_data)
        return batch_data


@torch.no_grad()
def test():
    torch.set_default_device("cpu")
    torch.set_num_threads(1)
    x = torch.randn(1, 481, 200, 2)
    batch_data = {"perturbed_spectra": x}
    net = CSMGAN_Wrapper()
    net.eval()
    print(net.net.bottleneck.receptive_field)
    net(batch_data)
    print(batch_data["fake_spectra"].shape)
    n_params = sum([p.numel() for p in net.parameters()])
    print(f"number of parameters: {n_params / 1e6}M")
    # calculate rtf
    import time

    start = time.time()
    for i in range(100):
        net(batch_data)
    end = time.time()
    print(f"RTF: {(end - start) / 100}")


if __name__ == "__main__":
    test()
