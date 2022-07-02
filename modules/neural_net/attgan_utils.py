# Copyright (C) 2018 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
#
# This work is licensed under the MIT License. To view a copy of this license,
# visit https://opensource.org/licenses/MIT.

"""Network components."""
import torch
import torch.nn as nn


class SwitchNorm1d(nn.Module):
    def __init__(
        self, num_features, eps=1e-5, momentum=0.997, using_moving_average=True
    ):
        super(SwitchNorm1d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.weight = nn.Parameter(torch.ones(1, num_features))
        self.bias = nn.Parameter(torch.zeros(1, num_features))
        self.mean_weight = nn.Parameter(torch.ones(2))
        self.var_weight = nn.Parameter(torch.ones(2))
        self.register_buffer("running_mean", torch.zeros(1, num_features))
        self.register_buffer("running_var", torch.zeros(1, num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.zero_()
        self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError("expected 2D input (got {}D input)".format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        mean_ln = x.mean(1, keepdim=True)
        var_ln = x.var(1, keepdim=True)

        if self.training:
            mean_bn = x.mean(0, keepdim=True)
            var_bn = x.var(0, keepdim=True)
            if self.using_moving_average:
                self.running_mean.mul_(self.momentum)
                self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                self.running_var.mul_(self.momentum)
                self.running_var.add_((1 - self.momentum) * var_bn.data)
            else:
                self.running_mean.add_(mean_bn.data)
                self.running_var.add_(mean_bn.data**2 + var_bn.data)
        else:
            mean_bn = torch.autograd.Variable(self.running_mean)
            var_bn = torch.autograd.Variable(self.running_var)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        mean = mean_weight[0] * mean_ln + mean_weight[1] * mean_bn
        var = var_weight[0] * var_ln + var_weight[1] * var_bn

        x = (x - mean) / (var + self.eps).sqrt()
        return x * self.weight + self.bias


class SwitchNorm2d(nn.Module):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.997,
        using_moving_average=True,
        using_bn=True,
        last_gamma=False,
    ):
        super(SwitchNorm2d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        if self.using_bn:
            self.mean_weight = nn.Parameter(torch.ones(3))
            self.var_weight = nn.Parameter(torch.ones(3))
        else:
            self.mean_weight = nn.Parameter(torch.ones(2))
            self.var_weight = nn.Parameter(torch.ones(2))
        if self.using_bn:
            self.register_buffer("running_mean", torch.zeros(1, num_features, 1))
            self.register_buffer("running_var", torch.zeros(1, num_features, 1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)

        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in**2
        var_ln = temp.mean(1, keepdim=True) - mean_ln**2

        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn**2
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.running_mean.add_(mean_bn.data)
                    self.running_var.add_(mean_bn.data**2 + var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(self.running_mean)
                var_bn = torch.autograd.Variable(self.running_var)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        if self.using_bn:
            mean = (
                mean_weight[0] * mean_in
                + mean_weight[1] * mean_ln
                + mean_weight[2] * mean_bn
            )
            var = (
                var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn
            )
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln

        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


def add_normalization_1d(layers, fn, n_out):
    if fn == "none":
        pass
    elif fn == "batchnorm":
        layers.append(nn.BatchNorm1d(n_out))
    elif fn == "instancenorm":
        layers.append(Unsqueeze(-1))
        layers.append(nn.InstanceNorm1d(n_out, affine=True))
        layers.append(Squeeze(-1))
    elif fn == "switchnorm":
        layers.append(SwitchNorm1d(n_out))
    else:
        raise Exception("Unsupported normalization: " + str(fn))
    return layers


def add_normalization_2d(layers, fn, n_out):
    if fn == "none":
        pass
    elif fn == "batchnorm":
        layers.append(nn.BatchNorm2d(n_out))
    elif fn == "instancenorm":
        layers.append(nn.InstanceNorm2d(n_out, affine=True))
    elif fn == "switchnorm":
        layers.append(SwitchNorm2d(n_out))
    else:
        raise Exception("Unsupported normalization: " + str(fn))
    return layers


def add_activation(layers, fn):
    if fn == "none":
        pass
    elif fn == "relu":
        layers.append(nn.ReLU())
    elif fn == "lrelu":
        layers.append(nn.LeakyReLU())
    elif fn == "sigmoid":
        layers.append(nn.Sigmoid())
    elif fn == "tanh":
        layers.append(nn.Tanh())
    else:
        raise Exception("Unsupported activation function: " + str(fn))
    return layers


class Squeeze(nn.Module):
    def __init__(self, dim):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)


class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


class LinearBlock(nn.Module):
    def __init__(self, n_in, n_out, norm_fn="none", acti_fn="none"):
        super(LinearBlock, self).__init__()
        layers = [nn.Linear(n_in, n_out, bias=(norm_fn == "none"))]
        layers = add_normalization_1d(layers, norm_fn, n_out)
        layers = add_activation(layers, acti_fn)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Conv2dBlock(nn.Module):
    def __init__(
        self, n_in, n_out, kernel_size, stride=1, padding=0, norm_fn=None, acti_fn=None
    ):
        super(Conv2dBlock, self).__init__()
        layers = [
            nn.Conv2d(
                n_in,
                n_out,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=(norm_fn == "none"),
            )
        ]
        layers = add_normalization_2d(layers, norm_fn, n_out)
        layers = add_activation(layers, acti_fn)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ConvTranspose2dBlock(nn.Module):
    def __init__(
        self, n_in, n_out, kernel_size, stride=1, padding=0, norm_fn=False, acti_fn=None
    ):
        super(ConvTranspose2dBlock, self).__init__()
        layers = [
            nn.ConvTranspose2d(
                n_in,
                n_out,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=(norm_fn == "none"),
            )
        ]
        layers = add_normalization_2d(layers, norm_fn, n_out)
        layers = add_activation(layers, acti_fn)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
