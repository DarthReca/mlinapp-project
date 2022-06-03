import pytorch_lightning as pl
import neural_net
import pytorch_lightning.callbacks as pl_call
from lit_data_module import CelebADataModule


def main():
    # Reproducibility
    pl.seed_everything(42, True)
    # Setup
    datamodule = CelebADataModule(8)
    model = neural_net.AttGAN({}, {"lr": 0.0002, "betas": (0.5, 0.999)})
    callbacks = [
        pl_call.RichModelSummary(),
        # pl_call.EarlyStopping(),
        pl_call.ModelCheckpoint(
            every_n_epochs=10, dirpath="checkpoints", monitor="loss", save_top_k=5
        ),
        pl_call.LearningRateMonitor(logging_interval="epoch"),
    ]
    trainer = pl.Trainer(accelerator="auto", callbacks=callbacks)
    # Train
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
