from typing import Dict, Any, Tuple

import pytorch_lightning as pl
import torch.optim
from torch import autograd
import torchmetrics as tm

from .attgan_parts import Discriminators, Generator

# Tutorial: https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/basic-gan.html


def extract_rgb(img: torch.Tensor) -> torch.Tensor:
    return (img * 255).round().byte()


def gradient_penalty(f, real, fake=None):
    def interpolate(a, b=None):
        device = torch.device('cuda:'+str(a.get_device()) if a.get_device()>=0 else 'cpu')
        if b is None:  # interpolation in DRAGAN
            beta = torch.rand_like(a, device=device)
            b = a + 0.5 * a.var().sqrt() * beta
        alpha = torch.rand(a.size(0), 1, 1, 1, device=device)
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
    """
    Parameters
    ----------
    lambda_rec: float
        Weight for reconstruction loss.
    lambda_gp: float
        Weight for gradient penalty.
    lambda_dc: float
        Weight for classification loss in discriminator training.
    lambda_gc: float
        Weight for classification loss in generator training.
    """

    def __init__(
        self,
        model_params: Dict[str, Any],
        optimizers_params: Dict[str, Any],
        target_attribute: int,
        thres_int=1,
        lambda_rec: float = 100.0,
        lambda_gp: float = 10.0,
        lambda_dc: float = 10.0,
        lambda_gc: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.generator = Generator(**model_params)
        self.discriminators = Discriminators(**model_params)

        weights = torch.load(f"weights/inject{self.generator.inject_layers}.pth",
                             "cuda" if torch.cuda.is_available() else "cpu")
        self.generator.load_state_dict(weights)

        self.reconstruction_loss = torch.nn.L1Loss()
        self.adversarial_loss = torch.nn.BCEWithLogitsLoss()
        self.discriminators_loss = torch.nn.BCEWithLogitsLoss()

        self.metrics = tm.MetricCollection([tm.image.InceptionScore()])

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

        idx = torch.randperm(len(att))
        desired_att = att[idx].contiguous()
        att_a_ = (att * 2 - 1) * self.hparams["thres_int"]
        att_b_ = (
            (desired_att * 2 - 1)
            * torch.rand_like(desired_att.float())
            * (2 * self.hparams["thres_int"])
        )

        # Train generator
        if optimizer_idx == 0:
            for p in self.discriminators.parameters():
                p.requires_grad = False
            # 1) The input images pass thorugh the encoder part, producing the latent vector zs_a
            zs_a = self.generator(img, mode="enc")
            # 2) The decoder gets as input the latent space and the conditioned attributes producing the fake image
            img_fake = self.generator(zs_a, att_b_, mode="dec")
            # 3) The decoder gets as input the latent space and the original attributes reconstructing the original image
            img_recon = self.generator(zs_a, att_a_, mode="dec")
            # 4) The discriminators (Discriminator and classifier) get as input the fake image and gives
            #    as output the choice between real/fake and the attributes classified by the classifiers
            d_fake, dc_fake = self.discriminators(img_fake)

            # Reconstruction loss
            r_loss = self.reconstruction_loss(img_recon, img)
            # Attribute Classification constraint
            d_loss = self.discriminators_loss(dc_fake, desired_att.float())
            # Adversarial loss (generator) -> how much the discriminator is been fooled predicting "real" when the images were actually fake
            a_loss = self.adversarial_loss(d_fake, torch.ones_like(d_fake)) 
            # Compute overall loss (generator)
            g_loss = (
                a_loss
                + self.hparams["lambda_gc"] * d_loss
                + self.hparams["lambda_rec"] * r_loss
            )
            self.log("generator_loss", g_loss)
            return g_loss

        # Train discriminator
        if optimizer_idx == 1:
            for p in self.discriminators.parameters():
                p.requires_grad = True

            # 1) The generator produces the fake images
            img_fake = self.generator(img, att_b_,  mode="enc-dec").detach()
            # 2) The discriminator gets as input the real images, saying if they are real/fake and predicting their attributes
            d_real, dc_real = self.discriminators(img)
            # 3) The discriminator gets as input the fake images, saying if they are real/fake and predicting their attributes
            d_fake, dc_fake = self.discriminators(img_fake)
            
            # Compute the discriminator adversarial loss
            a_loss = self.adversarial_loss(
                d_real, torch.ones_like(d_real) # saying that the d_real were supposed to be predicted as real
            ) + self.adversarial_loss(d_fake, torch.zeros_like(d_fake)) # saying that the d_fake were supposed to be predicted as fake
            # Compute the gradient penalty ??????????
            a_gp = gradient_penalty(self.discriminators, img)
            # Compute the discriminaotor loss (of classified attributes)
            dc_loss = self.discriminators_loss(dc_real, att.float())
            # Compute the overall loss
            d_loss = (
                a_loss
                + self.hparams["lambda_gp"] * a_gp
                + self.hparams["lambda_dc"] * dc_loss
            )
            self.log("discriminator_loss", d_loss)
            return d_loss

    def validation_step(self, batch, batch_idx: int):
        img, att = batch

        target = torch.zeros_like(att)
        target[:, self.hparams["target_attribute"]] = 1

        fake = (self.generator(img, target) * 255).round().byte()
        self.metrics.update(fake)
        for im in fake:
            self.logger.experiment.log_image(
                im, step=self.global_step, image_channels="first"
            )

    def validation_epoch_end(self, output) -> None:
        for k, v in self.metrics.compute().items():
            if isinstance(v, tuple):
                for i, single in enumerate(v):
                    self.log(k + f"_{i}", single)
            else:
                self.log(k, v)

    def test_step(self, batch, batch_idx: int):
        img, att = batch
        target = torch.zeros_like(att)
        target[0] = 1
        out = self.generator(img, target)
