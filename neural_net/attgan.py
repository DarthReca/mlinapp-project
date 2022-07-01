from typing import Dict, Any, Tuple

import pytorch_lightning as pl
import torch.optim
from torch import autograd
import torchmetrics as tm
from torchmetrics.image.inception import InceptionScore

from .attgan_parts import Discriminators, Generator

# Tutorial: https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/basic-gan.html


def extract_rgb(img: torch.Tensor) -> torch.Tensor:
    return (img * 255).round().byte()


def gradient_penalty(f, real, fake=None):
    def interpolate(a, b=None):
        device = torch.device(
            "cuda:" + str(a.get_device()) if a.get_device() >= 0 else "cpu"
        )
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
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        # Define the AttGan Generator
        self.generator = Generator(
            args.enc_dim,
            args.enc_layers,
            args.enc_norm,
            args.enc_acti,
            args.dec_dim,
            args.dec_layers,
            args.dec_norm,
            args.dec_acti,
            args.n_attrs,
            args.shortcut_layers,
            args.inject_layers,
            args.img_size,
        )

        # Define the AttGan Discriminator
        self.discriminators = Discriminators(
            args.dis_dim,
            args.dis_norm,
            args.dis_acti,
            args.dis_fc_dim,
            args.dis_fc_norm,
            args.dis_fc_acti,
            args.dis_layers,
            args.img_size,
        )

        # Load the initial weights
        weights = torch.load(
            f"weights/inject{self.generator.inject_layers}.pth",
            "cuda" if torch.cuda.is_available() else "cpu",
        )

        try:
            self.generator.load_state_dict(weights)
        except RuntimeError:
            self.generator.load_state_dict(weights["G"])

        # Define the losses
        self.reconstruction_loss = torch.nn.L1Loss()
        self.adversarial_loss = torch.nn.BCEWithLogitsLoss()
        self.discriminators_loss = torch.nn.BCEWithLogitsLoss()

        # Define weights of losses
        self.lambda_rec = args.lambda_1  # Weight for reconstruction loss
        self.lambda_gc = (
            args.lambda_2
        )  # Weight for classification loss in generator training.
        self.lambda_dc = (
            args.lambda_3
        )  # Weight for classification loss in discriminator training.
        self.lambda_gp = args.lambda_gp  # Weight for gradient penalty.

        # Define optimizer hyperparameters
        self.lr = args.lr
        self.betas = args.betas

        # Define metrics
        self.metrics = tm.MetricCollection([InceptionScore()])

        # Define target attribute index
        self.target_attribute_index = args.target_attr_index
        self.thres_int = args.thres_int

        # Define b distribution
        self.b_distribution = args.b_distribution

    def configure_optimizers(self):
        gen_optim = torch.optim.Adam(
            self.generator.parameters(), lr=self.lr, betas=self.betas
        )
        disc_optim = torch.optim.Adam(
            self.discriminators.parameters(), lr=self.lr, betas=self.betas
        )
        return gen_optim, disc_optim

    def training_step(
        self, batch, batch_idx: int, optimizer_idx: int
    ) -> Dict[str, float]:

        # 1 batch of images with associated labels (with values 0/1)
        img_a, att_a = batch

        permuted_indexes = torch.randperm(len(att_a))
        att_b = att_a[permuted_indexes].contiguous()  # desired attributes

        att_b = att_b.float()
        att_a = att_a.float()

        att_a_ = (att_a * 2 - 1) * self.thres_int  # att_a shifted to -0.5,0.5

        if self.b_distribution == "none":
            att_b_ = (att_b * 2 - 1) * self.thres_int

        if self.b_distribution == "uniform":
            att_b_ = (att_b * 2 - 1) * torch.rand_like(att_b) * (2 * self.thres_int)

        if self.b_distribution == "truncated_normal":
            att_b_ = (
                (att_b * 2 - 1)
                * (torch.fmod(torch.randn_like(att_b), 2) + 2)
                / 4.0
                * (2 * self.thres_int)
            )

        # Train generator
        if optimizer_idx == 0:
            for p in self.discriminators.parameters():
                p.requires_grad = False
            # 1) The input images pass thorugh the encoder part, producing the latent vector zs_a
            zs_a = self.generator(img_a, mode="enc")
            # 2) The decoder gets as input the latent space and the conditioned attributes producing the fake image
            img_fake = self.generator(zs_a, att_b_, mode="dec")
            # 3) The decoder gets as input the latent space and the original attributes reconstructing the original image
            img_recon = self.generator(zs_a, att_a_, mode="dec")
            # 4) The discriminators (Discriminator and classifier) get as input the fake image and gives
            #    as output the choice between real/fake and the attributes classified by the classifiers
            d_fake, dc_fake = self.discriminators(img_fake)

            # Reconstruction loss
            r_loss = self.reconstruction_loss(img_recon, img_a)
            self.log("reconstruction_loss", r_loss)
            # Attribute Classification constraint
            d_loss = self.discriminators_loss(dc_fake, att_b.float())
            # Adversarial loss (generator) -> how much the discriminator is been fooled predicting "real" when the images were actually fake
            a_loss = self.adversarial_loss(d_fake, torch.ones_like(d_fake))
            # Compute overall loss (generator)
            g_loss = a_loss + self.lambda_gc * d_loss + self.lambda_rec * r_loss
            self.log("generator_loss", g_loss)
            return g_loss

        # Train discriminator
        if optimizer_idx == 1:
            for p in self.discriminators.parameters():
                p.requires_grad = True

            # 1) The generator produces the fake images
            img_fake = self.generator(img_a, att_b_, mode="enc-dec").detach()
            # 2) The discriminator gets as input the real images, saying if they are real/fake and predicting their attributes
            d_real, dc_real = self.discriminators(img_a)
            # 3) The discriminator gets as input the fake images, saying if they are real/fake and predicting their attributes
            d_fake, dc_fake = self.discriminators(img_fake)

            # Compute the discriminator adversarial loss
            a_loss = self.adversarial_loss(
                d_real,
                torch.ones_like(
                    d_real
                ),  # saying that the d_real were supposed to be predicted as real
            ) + self.adversarial_loss(
                d_fake, torch.zeros_like(d_fake)
            )  # saying that the d_fake were supposed to be predicted as fake
            # Compute the gradient penalty ??????????
            a_gp = gradient_penalty(self.discriminators, img_a)
            # Compute the discriminaotor loss (of classified attributes)
            dc_loss = self.discriminators_loss(dc_real, att_a)
            # Compute the overall loss
            d_loss = a_loss + self.lambda_gp * a_gp + self.lambda_dc * dc_loss
            self.log("discriminator_loss", d_loss)
            return d_loss

    def validation_step(self, batch, batch_idx: int):
        # TODO set no_beard to 0 ?
        img, att = batch

        target = torch.zeros_like(att)
        target[:, self.target_attribute_index] = 1

        fake = (self.generator(img, target) * 255).round().byte()
        self.metrics.update(fake)
        for i, im in enumerate(fake):
            self.logger.experiment.log_image(
                im.cpu(), step=self.global_step, image_channels="first", name=f"Image-{i}"
            )

    def validation_epoch_end(self, output) -> None:
        for k, v in self.metrics.compute().items():
            if isinstance(v, tuple):
                for i, single in enumerate(v):
                    self.log(k + f"_{i}", single)
            else:
                self.log(k, v)

    def test_step(self, batch, batch_idx: int):
        # TODO set no_beard to 0 ?
        img, att = batch
        
        target = torch.zeros_like(att)
        target[:, self.target_attribute_index] = 1
        
        out = self.generator(img, target)
