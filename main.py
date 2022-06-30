import datetime
import comet_ml as cml
import pytorch_lightning as pl
import neural_net
import pytorch_lightning.callbacks as pl_call
from dataset.lit_data_module import CelebADataModule
import pytorch_lightning.loggers as loggers
import argparse

# Define a subset of attributes we want to use from the ones of CelebA.
attrs_default = [
    "Bald",
    "Bangs",
    "Black_Hair",
    "Blond_Hair",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Eyeglasses",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "No_Beard",
    "Pale_Skin",
    "Young",
]


def parse(args=None):

    parser = argparse.ArgumentParser()
    # Attributes
    parser.add_argument(
        "--attrs",
        dest="attrs",
        default=attrs_default,
        nargs="+",
        help="attributes to learn",
    )
    parser.add_argument(
        "--target_attr", dest="target_attr", default="Mustache", help="Target attribute"
    )

    # Data
    parser.add_argument("--data", dest="data", type=str, default="CelebA")
    parser.add_argument("--img_size", dest="img_size", type=int, default=128)
    # Paths
    parser.add_argument(
        "--data_path", dest="data_path", type=str, default="data/small_dataset"
    )  #'data/img_align_celeba'
    parser.add_argument(
        "--attr_path", dest="attr_path", type=str, default="data/list_attr_small.txt"
    )  #'data/list_attr_celeba.txt'

    parser.add_argument(
        "--indices_path",
        dest="indices_path",
        type=str,
        default="dataset/indices_test.npy",
        help="numpy file with indices of the considered images during training",
    )

    # Network
    parser.add_argument(
        "--shortcut_layers", dest="shortcut_layers", type=int, default=1
    )
    parser.add_argument("--inject_layers", dest="inject_layers", type=int, default=0)
    parser.add_argument("--enc_dim", dest="enc_dim", type=int, default=64)
    parser.add_argument("--dec_dim", dest="dec_dim", type=int, default=64)
    parser.add_argument("--dis_dim", dest="dis_dim", type=int, default=64)
    parser.add_argument("--dis_fc_dim", dest="dis_fc_dim", type=int, default=1024)
    parser.add_argument("--enc_layers", dest="enc_layers", type=int, default=5)
    parser.add_argument("--dec_layers", dest="dec_layers", type=int, default=5)
    parser.add_argument("--dis_layers", dest="dis_layers", type=int, default=5)
    parser.add_argument("--enc_norm", dest="enc_norm", type=str, default="batchnorm")
    parser.add_argument("--dec_norm", dest="dec_norm", type=str, default="batchnorm")
    parser.add_argument("--dis_norm", dest="dis_norm", type=str, default="instancenorm")
    parser.add_argument("--dis_fc_norm", dest="dis_fc_norm", type=str, default="none")
    parser.add_argument("--enc_acti", dest="enc_acti", type=str, default="lrelu")
    parser.add_argument("--dec_acti", dest="dec_acti", type=str, default="relu")
    parser.add_argument("--dis_acti", dest="dis_acti", type=str, default="lrelu")
    parser.add_argument("--dis_fc_acti", dest="dis_fc_acti", type=str, default="relu")
    parser.add_argument("--lambda_1", dest="lambda_1", type=float, default=100.0)
    parser.add_argument("--lambda_2", dest="lambda_2", type=float, default=10.0)
    parser.add_argument("--lambda_3", dest="lambda_3", type=float, default=1.0)
    parser.add_argument("--lambda_gp", dest="lambda_gp", type=float, default=10.0)

    parser.add_argument(
        "--mode", dest="mode", default="wgan", choices=["wgan", "lsgan", "dcgan"]
    )

    # Training
    parser.add_argument(
        "--epochs", dest="epochs", type=int, default=200, help="# of epochs"
    )
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=5)  # 32
    parser.add_argument("--num_workers", dest="num_workers", type=int, default=0)

    parser.add_argument(
        "--lr", dest="lr", type=float, default=0.0002, help="learning rate"
    )
    parser.add_argument("--beta1", dest="beta1", type=float, default=0.5)
    parser.add_argument("--beta2", dest="beta2", type=float, default=0.999)
    parser.add_argument(
        "--n_d", dest="n_d", type=int, default=5, help="# of d updates per g update"
    )

    parser.add_argument(
        "--b_distribution",
        dest="b_distribution",
        default="none",
        choices=["none", "uniform", "truncated_normal"],
    )
    parser.add_argument("--thres_int", dest="thres_int", type=float, default=0.5)
    parser.add_argument("--test_int", dest="test_int", type=float, default=1.0)

    parser.add_argument(
        "--n_samples", dest="n_samples", type=int, default=16, help="# of sample images"
    )

    # Saving params
    parser.add_argument("--save_interval", dest="save_interval", type=int, default=1000)
    parser.add_argument(
        "--sample_interval",
        dest="sample_interval",
        type=int,
        default=1,
        help="# of iteration between evaluations",
    )  # 1000
    parser.add_argument("--gpu", dest="gpu", action="store_true")
    parser.add_argument("--multi_gpu", dest="multi_gpu", action="store_true")
    parser.add_argument(
        "--experiment_name",
        dest="experiment_name",
        default=datetime.datetime.now().strftime("%I%M%p%B%d%Y"),
    )

    return parser.parse_args(args)


def main():
    # Get arguments
    args = parse()

    args.lr_base = args.lr
    args.betas = (args.beta1, args.beta2)

    args.n_attrs = len(args.attrs)  # 13 (default)
    args.target_attr_index = args.attrs.index(args.target_attr)

    # Reproducibility
    pl.seed_everything(42, True)

    # Setup data module
    datamodule = CelebADataModule(
        selected_attrs=attrs_default,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.img_size,
        indices_file=args.indices_path,
        data_path=args.data_path,
        attr_path=args.attr_path,
    )
    # Setup model
    model = neural_net.AttGAN(args)

    # Setup trainer
    callbacks = [
        pl_call.RichModelSummary(),
        pl_call.RichProgressBar(),
        pl_call.EarlyStopping(
            monitor="generator_loss", patience=2, min_delta=0.001, verbose=True
        ),
        pl_call.ModelCheckpoint(
            every_n_epochs=3,
            dirpath="checkpoints",
            monitor="generator_loss",
            save_top_k=2,
            verbose=True,
        ),
        pl_call.LearningRateMonitor(logging_interval="epoch"),
        pl_call.Timer("00:01:50:00")
    ]
    # Setup Comet logger
    logger = loggers.CometLogger(
        api_key="sB2qT71Uklji1EGNlkZ2WhuzL", project_name="mlinapp"
    )

    trainer = pl.Trainer(
        accelerator="auto",
        callbacks=callbacks,
        logger=logger,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        log_every_n_steps=20,
        max_epochs=args.epochs,
    )
    # Train
    trainer.fit(model, datamodule=datamodule)
    # Ending
    logger.experiment.log_model(
        "Best model", trainer.checkpoint_callback.best_model_path
    )
    logger.experiment.end()


if __name__ == "__main__":
    main()
