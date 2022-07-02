import datetime
import time
import comet_ml as cml
import pytorch_lightning as pl
import pytorch_lightning.callbacks as pl_call
from modules.neural_net.attgan import AttGAN
from modules.dataset.celeba_data_module import CelebADataModule
import pytorch_lightning.loggers as loggers
import argparse
from modules import utils
from modules.utils import bcolors, pretty_time_delta

# Define a subset of attributes we want to use from the ones of CelebA.
# These are the 13 attributes used in the original paper
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
    "Young"
]

# These are 20 attributes including the default 13
attrs_default_plus = [
    "Bald",
    "Bangs",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Eyeglasses",
    "Gray_Hair",
    "Heavy_Makeup",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "No_Beard",
    "Pale_Skin",
    "Pointy_Nose",
    "Smiling",
    "Wearing_Hat",
    "Wearing_Lipstick"
    "Young"
]


def parse_args():
    parser = argparse.ArgumentParser()

    # which attributes to consider
    parser.add_argument(
        "--attrs", dest="attrs", default="default", choices=["default", "plus", "custom"], help="which attributes to consider among the proposed ones"
    )
    parser.add_argument(
        "--attrs_list",
        dest="attrs_list",
        default=None,
        nargs="+",
        help="list of attributes to learn, must only be specified when using '--attrs=custom'"
    )
    # which attribute will be forced to 1
    parser.add_argument(
        "--target_attr", dest="target_attr", default="Mustache", help="target attribute that will be forced to 1"
    )

    # images dimensions
    parser.add_argument("--img_size", dest="img_size", type=int,
                        default=128, help="dimensions in pixel of the images' side")
    # data path
    parser.add_argument(
        "--data_root", dest="data_root", type=str, default="data", help="where to find the dataset"
    )
    # indices path
    parser.add_argument(
        "--indices_path",
        dest="indices_path",
        type=str,
        default="data/chosen_indices.npy",
        help="numpy file with indices of the considered subset",
    )

    # how many shortcut layers to use
    parser.add_argument(
        "--shortcut_layers", dest="shortcut_layers", type=int, default=1
    )
    # various dimensions
    parser.add_argument("--enc_dim", dest="enc_dim", type=int, default=64)
    parser.add_argument("--dec_dim", dest="dec_dim", type=int, default=64)
    parser.add_argument("--dis_dim", dest="dis_dim", type=int, default=64)
    parser.add_argument("--dis_fc_dim", dest="dis_fc_dim",
                        type=int, default=1024)
    # number of layers
    parser.add_argument("--enc_layers", dest="enc_layers", type=int, default=5)
    parser.add_argument("--dec_layers", dest="dec_layers", type=int, default=5)
    parser.add_argument("--dis_layers", dest="dis_layers", type=int, default=5)
    # normalization layers type
    parser.add_argument("--enc_norm", dest="enc_norm",
                        type=str, default="batchnorm")
    parser.add_argument("--dec_norm", dest="dec_norm",
                        type=str, default="batchnorm")
    parser.add_argument("--dis_norm", dest="dis_norm",
                        type=str, default="instancenorm")
    parser.add_argument("--dis_fc_norm", dest="dis_fc_norm",
                        type=str, default="none")
    # activation layers type
    parser.add_argument("--enc_acti", dest="enc_acti",
                        type=str, default="lrelu")
    parser.add_argument("--dec_acti", dest="dec_acti",
                        type=str, default="relu")
    parser.add_argument("--dis_acti", dest="dis_acti",
                        type=str, default="lrelu")
    parser.add_argument("--dis_fc_acti", dest="dis_fc_acti",
                        type=str, default="relu")
    # weight of each loss in the final loss function
    parser.add_argument("--lambda_1", dest="lambda_1",
                        type=float, default=100.0)
    parser.add_argument("--lambda_2", dest="lambda_2",
                        type=float, default=10.0)
    parser.add_argument("--lambda_3", dest="lambda_3", type=float, default=1.0)
    parser.add_argument("--lambda_gp", dest="lambda_gp",
                        type=float, default=10.0)

    # training stuff
    parser.add_argument(
        "--epochs", dest="epochs", type=int, default=20, help="number of epochs"
    )
    parser.add_argument("--batch_size", dest="batch_size",
                        type=int, default=32)
    parser.add_argument("--num_workers", dest="num_workers",
                        type=int, default=2)
    parser.add_argument(
        "--training_approach", dest="training_approach", default="mustache", choices=["mustache", "generic"]
    )
    parser.add_argument(
        "--lr", dest="lr", type=float, default=0.0001, help="starting learning rate"
    )
    parser.add_argument("--beta1", dest="beta1", type=float, default=0.5)
    parser.add_argument("--beta2", dest="beta2", type=float, default=0.999)
    parser.add_argument(
        "--dg_ratio", dest="dg_ratio", type=int, default=5, help="# of d updates per g update"
    )
    parser.add_argument("--no_pretrained",
                        dest="no_pretrained", action="store_true")

    # how many images to infer during validation steps
    parser.add_argument(
        "--val_samples", dest="val_samples", type=int, default=12, help="number of sample images in validation"
    )

    # saving state
    parser.add_argument("--upload_weights",
                        dest="upload_weights", action="store_true", help="upload final weights to comet")
    parser.add_argument("--log_interval",
                        dest="log_interval", type=int, default=50, help="number of steps between logs")
    parser.add_argument(
        "--val_interval",
        dest="val_interval",
        type=int,
        default=1,
        help="number of epochs between evaluation steps",
    )
    parser.add_argument("--force_cpu", dest="force_cpu", action="store_true")
    parser.add_argument(
        "--experiment_name",
        dest="experiment_name",
        default=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )

    return parser.parse_args()


def main():
    start_time = time.time()

    # Get arguments
    args = parse_args()

    if args.attrs == "default":
        args.attrs_list = attrs_default
    elif args.attrs == "plus":
        args.attrs_list = attrs_default_plus
    elif args.attrs == "custom":
        if args.attrs_list is None:
            print(
                f"{bcolors.ERROR}You specified '--attrs=custom' but left '--attrs_list' empty!{bcolors.ENDC}")
            raise SystemExit
    args.n_attrs = len(args.attrs_list)
    try:
        args.target_attr_index = args.attrs_list.index(args.target_attr)
    except ValueError:
        print(f"{bcolors.ERROR}The specified '--target_attr={args.target_attr}' is not among the currently considered attributes!{bcolors.ENDC}")
        raise SystemExit
    args.betas = (args.beta1, args.beta2)

    print("Training on the following attributes")
    print(args.attrs_list)

    # Reproducibility
    pl.seed_everything(42, True)

    # Setup data module
    celeba_datamodule = CelebADataModule(
        selected_attrs=args.attrs_list,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        indices_file=args.indices_path,
        data_root=args.data_root,
        num_val_samples=args.val_samples
    )

    # Setup model
    model = AttGAN(args)

    # Setup trainer
    callbacks = [
        pl_call.RichModelSummary(),
        # pl_call.RichProgressBar(leave=True),  # per qualche motivo a me non va -db
        pl_call.TQDMProgressBar(),
        pl_call.EarlyStopping(
            monitor="generator_loss", patience=20, min_delta=0.001, verbose=True
        ),
        pl_call.ModelCheckpoint(
            every_n_epochs=1,
            dirpath="checkpoints",
            monitor="generator_loss",
            save_top_k=2,
            verbose=True,
        ),
        pl_call.LearningRateMonitor(logging_interval="epoch")
    ]

    # Setup Comet logger
    logger = loggers.CometLogger(
        api_key="TvZ83pu3DEe7ETHo5gOp49GAg", project_name="mlinapp-project", experiment_name=args.experiment_name
    )

    # Setup trainer
    trainer = pl.Trainer(
        accelerator="cpu" if args.force_cpu else "auto",
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=args.val_interval,
        log_every_n_steps=args.log_interval,
        max_epochs=args.epochs
    )

    # Train
    trainer.fit(model, datamodule=celeba_datamodule)

    # Ending
    if args.upload_weights:
        logger.experiment.log_model(
            "Best model", trainer.checkpoint_callback.best_model_path
        )
    logger.experiment.end()
    end_time = time.time()
    print(f"Running the entire script took",
          pretty_time_delta(end_time-start_time))


if __name__ == "__main__":
    main()
