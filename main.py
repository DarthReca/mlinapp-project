import comet_ml as cml
import pytorch_lightning as pl
import neural_net
import pytorch_lightning.callbacks as pl_call
from lit_data_module import CelebADataModule
import pytorch_lightning.loggers as loggers


def main():
    # Reproducibility
    pl.seed_everything(42, True)
    # Setup model and data
    datamodule = CelebADataModule(8)
    model = neural_net.AttGAN({"n_attrs": 40}, {"lr": 0.0002, "betas": (0.5, 0.999)})
    # Setup trainer
    callbacks = [
        pl_call.RichModelSummary(),
        pl_call.EarlyStopping(monitor="loss", patience=50, min_delta=0.001),
        pl_call.ModelCheckpoint(
            every_n_epochs=10, dirpath="checkpoints", monitor="loss", save_top_k=5
        ),
        pl_call.LearningRateMonitor(logging_interval="epoch"),
    ]
    logger = loggers.CometLogger(
        api_key="sB2qT71Uklji1EGNlkZ2WhuzL", project_name="mlinapp"
    )
    trainer = pl.Trainer(accelerator="auto", callbacks=callbacks, logger=logger)
    # Train
    trainer.fit(model, datamodule=datamodule)
    # Ending
    logger.experiment.log_model(
        "Best model", trainer.checkpoint_callback.best_model_path
    )
    logger.experiment.end()


if __name__ == "__main__":
    main()
