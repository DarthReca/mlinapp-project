from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch.optim
import torchmetrics as tm
from torch import autograd
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

from .attgan_parts import Discriminators, Generator

# Tutorial: https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/basic-gan.html


def images_from_tensor(img: torch.Tensor) -> torch.Tensor:
    img = img.clone().detach()
    img = img.clamp_(-1, 1).sub_(-1).div(2)
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
        if (not args.no_pretrained) and (not args.resume_from_path):
            # Load the initial weights
            weights = torch.load(
                "weights/pretrained.pth",
                "cuda" if (torch.cuda.is_available() and not args.force_cpu) else "cpu",
            )

            try:
                self.load_state_dict(weights)
                print("Preloaded weights for the entire net")
            except RuntimeError:
                self.generator.load_state_dict(weights["G"])
                print("Preloaded weights for the generator only")

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
        # self.metrics = tm.MetricCollection([InceptionScore()])
        # self.fid = FrechetInceptionDistance(feature=64, reset_real_features=False)
        self.accuracy = tm.Accuracy()

        # Define target attribute index
        self.target_attribute_index = args.target_attr_index

        # Define training approach
        self.training_approach = args.training_approach
        self.dg_ratio = args.dg_ratio

        # FreezeD
        for p in self.discriminators.conv[: args.freeze_layers].parameters():
            p.requires_grad = False

    def configure_optimizers(self):
        gen_optim = torch.optim.Adam(
            self.generator.parameters(), lr=self.lr, betas=self.betas
        )
        disc_optim = torch.optim.Adam(
            self.discriminators.parameters(), lr=self.lr, betas=self.betas
        )
        return (
            {"optimizer": gen_optim, "frequency": 1},
            {"optimizer": disc_optim, "frequency": self.dg_ratio},
        )

    # LR warmup
    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        # generator
        if optimizer_idx == 0:
            gen_target_step = 800.0  # don't forget the dot here
            if self.trainer.global_step < gen_target_step:
                lr_scale = min(
                    1.0, float(self.trainer.global_step + 1) / gen_target_step
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_scale * self.lr
            optimizer.step(closure=optimizer_closure)

        # discriminator
        if optimizer_idx == 1:
            dis_target_step = 200.0  # don't forget the dot here
            if self.trainer.global_step < dis_target_step:
                lr_scale = min(
                    1.0, float(self.trainer.global_step + 1) / dis_target_step
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_scale * self.lr
            optimizer.step(closure=optimizer_closure)

    def training_step(
        self, batch, batch_idx: int, optimizer_idx: int
    ) -> Dict[str, float]:

        # 1 batch of images with associated labels (with values 0/1)
        orig_images_a, orig_attributes_a = batch

        orig_images = images_from_tensor(orig_images_a)
        self.fid.update(orig_images, real=True)

        # define how attributes should be conditioned
        if self.training_approach == "specific":
            fake_attributes_b = orig_attributes_a.clone().detach()
            for atts in fake_attributes_b:
                # Invert target attribute
                atts[self.target_attribute_index] = (
                    0 if atts[self.target_attribute_index] else 1
                )
        else:
            permuted_indexes = torch.randperm(len(orig_attributes_a))
            fake_attributes_b = orig_attributes_a[
                permuted_indexes
            ].contiguous()  # fake attributes

        fake_attributes_b = fake_attributes_b.float()
        orig_attributes_a = orig_attributes_a.float()

        shifted_orig_attributes_a_tilde = (
            orig_attributes_a * 2 - 1
        ) * 0.5  # orig_attributes_a shifted to -0.5,0.5

        shifted_fake_attributes_b_tilde = (
            fake_attributes_b * 2 - 1
        ) * 0.5  # orig_attributes_a shifted to -0.5,0.5

        # Train generator
        if optimizer_idx == 0:
            for p in self.discriminators.parameters():
                p.requires_grad = False
            # 1) The input images pass through the encoder part, producing the latent vector embedding_zs_a
            embedding_zs_a = self.generator(orig_images_a, mode="enc")
            # 2) The decoder gets as input the latent space and the conditioned attributes producing the fake image
            fake_images = self.generator(
                embedding_zs_a, shifted_fake_attributes_b_tilde, mode="dec"
            )
            # 3) The decoder gets as input the latent space and the orig attributes reconstructing the orig image
            reconstructed_images = self.generator(
                embedding_zs_a, shifted_orig_attributes_a_tilde, mode="dec"
            )
            # 4) The discriminators (Discriminator and classifier) get as input the fake image and gives
            #    as output the choice between real/fake and the attributes classified by the classifiers
            fakes_discrimination, fakes_classification = self.discriminators(
                fake_images
            )

            # Reconstruction loss
            r_loss = self.reconstruction_loss(reconstructed_images, orig_images_a)
            self.log("reconstruction_loss", r_loss)
            # Attribute Classification constraint
            d_loss = self.discriminators_loss(
                fakes_classification, fake_attributes_b.float()
            )
            # Adversarial loss (generator) -> how much the discriminator is been fooled predicting "real" when the images were actually fake
            a_loss = self.adversarial_loss(
                fakes_discrimination, torch.ones_like(fakes_discrimination)
            )
            # Compute overall loss (generator)
            g_loss = a_loss + self.lambda_gc * d_loss + self.lambda_rec * r_loss
            self.log("generator_loss", g_loss)
            return g_loss

        # Train discriminator
        if optimizer_idx == 1:
            for p in self.discriminators.parameters():
                p.requires_grad = True

            # 1) The generator produces the fake images
            fake_images = self.generator(
                orig_images_a, shifted_fake_attributes_b_tilde, mode="enc-dec"
            ).detach()
            # 2) The discriminator gets as input the real images, saying if they are real/fake and predicting their attributes
            reals_discrimination, reals_classification = self.discriminators(
                orig_images_a
            )
            # 3) The discriminator gets as input the fake images, saying if they are real/fake and predicting their attributes
            fakes_discrimination, fakes_classification = self.discriminators(
                fake_images
            )

            # Compute the discriminator adversarial loss
            all_ones = torch.ones_like(reals_discrimination)
            all_zeros = torch.zeros_like(fakes_discrimination)

            # log discriminator accuracy metric
            self.accuracy(torch.sigmoid(reals_discrimination), all_ones.int())
            self.accuracy(torch.sigmoid(fakes_discrimination), all_zeros.int())
            self.log("adversarial_accuracy", self.accuracy)

            a_loss = self.adversarial_loss(
                reals_discrimination,
                all_ones,  # saying that the reals_discrimination were supposed to be predicted as real
            ) + self.adversarial_loss(
                fakes_discrimination, all_zeros
            )  # saying that the fakes_discrimination were supposed to be predicted as fake

            # Compute the gradient penalty
            a_gp = gradient_penalty(self.discriminators, orig_images_a)

            # Compute the discriminator loss (of classified attributes)
            dc_loss = self.discriminators_loss(reals_classification, orig_attributes_a)

            self.log("disc_classification_loss", d_loss)

            # Compute the overall loss
            d_loss = a_loss + self.lambda_gp * a_gp + self.lambda_dc * dc_loss
            self.log("discriminator_loss", d_loss)
            return d_loss

    def validation_step(self, batch, batch_idx: int):
        # this is target-specific!
        orig_images, orig_attributes = batch

        target = orig_attributes.clone().detach()
        target[:, self.target_attribute_index] = 1
        # target[:, self.target_attribute_index+1] = 0

        target = target.float()
        target = (target * 2 - 1) * 0.5  # target shifted to -0.5,0.5

        fake = self.generator(orig_images, target)

        fake = images_from_tensor(fake)
        orig_images = images_from_tensor(orig_images)

        # self.fid.update(orig_images, real=True)
        self.fid.update(fake, real=False)
        for i, (im, orig) in enumerate(zip(fake, orig_images)):
            self.logger.experiment.log_image(
                torch.cat([orig.cpu(), im.cpu()], dim=2),
                step=self.global_step,
                image_channels="first",
                name=f"Image-{i}",
            )

    def validation_epoch_end(self, output) -> None:
        # for k, v in self.metrics.compute().items():
        #     if isinstance(v, tuple):
        #         for i, single in enumerate(v):
        #             self.log(k + f"_{i}", single)
        #     else:
        #         self.log(k, v)
        # self.metrics.reset()

        self.log("FID", self.fid.compute().item())
        self.fid.reset()

    def test_step(self, batch, batch_idx: int):
        # this is target-specific!
        orig_images, orig_attributes = batch

        target = orig_attributes.clone().detach()
        target[:, self.target_attribute_index] = 1
        # target[:, self.target_attribute_index+1] = 0

        target = target.float()
        target = (target * 2 - 1) * 0.5  # target shifted to -0.5,0.5

        out = self.generator(orig_images, target)
