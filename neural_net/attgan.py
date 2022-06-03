from typing import Dict, Any

import pytorch_lightning as pl
import torch.optim
from torch import autograd

from .attgan_parts import Discriminators, Generator

# Tutorial: https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/basic-gan.html


def gradient_penalty(f, real, fake=None):
    def interpolate(a, b=None):
        if b is None:  # interpolation in DRAGAN
            beta = torch.rand_like(a)
            b = a + 0.5 * a.var().sqrt() * beta
        alpha = torch.rand(a.size(0), 1, 1, 1)
        inter = a + alpha * (b - a)
        return inter

    x = interpolate(real, fake).requires_grad_(True)
    pred = f(x)
    if isinstance(pred, tuple):
        pred = pred[0]
    grad = autograd.grad(
        outputs=pred,
        inputs=x,
        grad_outputs=torch.ones_like(pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad = grad.view(grad.size(0), -1)
    norm = grad.norm(2, dim=1)
    gp = ((norm - 1.0) ** 2).mean()
    return gp


class AttGAN(pl.LightningModule):
    def __init__(
        self,
        model_params: Dict[str, Any],
        optimizers_params: Dict[str, Any],
    ):
        super().__init__()
        self.save_hyperparameters()

        self.generator = Generator(**model_params)
        self.discriminators = Discriminators(**model_params)

        self.reconstruction_loss = torch.nn.L1Loss()
        self.adversarial_loss = torch.nn.BCEWithLogitsLoss()
        self.discriminators_loss = torch.nn.BCEWithLogitsLoss()

    def configure_optimizers(self):
        gen_optim = torch.optim.Adam(
            self.generator.parameters(), **self.hparams["optimizers_params"]
        )
        disc_optim = torch.optim.Adam(
            self.discriminators.parameters(), **self.hparams["optimizers_params"]
        )
        return gen_optim, disc_optim

    def training_step(
        self, batch, batch_idx: int, optimizer_idx: int
    ) -> Dict[str, float]:
        img, att = batch

        # TODO: Indagini più accurate su thres_int e questa generazione random di attributi
        idx = torch.randperm(len(att))
        desired_att = att[idx].contiguous()
        att_a_ = (att * 2 - 1) * 1.0  # args.thres_int
        att_b_ = (
            (desired_att * 2 - 1) * torch.rand_like(desired_att.float()) * (2 * 1.0)
        )  # args.thres_int)
        ####

        # Train generator
        if optimizer_idx == 0:
            zs_a = self.generator(img, mode="enc")
            img_fake = self.generator(zs_a, att_b_, mode="dec")
            img_recon = self.generator(zs_a, att_a_, mode="dec")
            d_fake, dc_fake = self.discriminators(img_fake)
            # TODO: protrebbe non essere uguale a freezare il discriminator (Indagine necessaria)
            d_fake, dc_fake = d_fake.detach(), dc_fake.detach()

            r_loss = self.reconstruction_loss(img_recon, img)
            d_loss = self.discriminators_loss(dc_fake, desired_att.float())
            a_loss = self.adversarial_loss(d_fake, torch.ones_like(d_fake))
            g_loss = a_loss + 10.0 * d_loss + 100.0 * r_loss

            return {
                "loss": g_loss,
                "adversarial_loss": a_loss,
                "discriminators_loss": d_loss,
                "reconstruction_loss": r_loss,
            }

        # Train discriminator
        if optimizer_idx == 1:
            img_fake = self.generator(img, att_b_).detach()
            d_real, dc_real = self.discriminators(img)
            d_fake, dc_fake = self.discriminators(img_fake)

            a_loss = self.adversarial_loss(
                d_real, torch.ones_like(d_real)
            ) + self.adversarial_loss(d_fake, torch.zeros_like(d_fake))
            a_gp = gradient_penalty(self.discriminators, img)
            dc_loss = self.discriminators_loss(dc_real, att.float())
            d_loss = a_loss + 10.0 * a_gp + 1.0 * dc_loss

            return {
                "loss": d_loss,
                "adversarial_loss": a_loss,
                "discriminators_loss": d_loss,
                "gradient_penalty": a_gp,
            }

    def test_step(self, batch, batch_idx: int):
        img, att = batch
        target = torch.zeros_like(att)
        target[0] = 1
        out = self.generator(img, target)
