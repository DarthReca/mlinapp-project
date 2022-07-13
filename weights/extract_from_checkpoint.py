import torch
import re
import argparse

def extract_from_checkpoint(input_checkpoint_path, output_weights_path):
    checkpoint = torch.load(input_checkpoint_path, map_location="cpu")
    all_weights = checkpoint['state_dict']
    weights_without_fid = {k:all_weights[k] for k in all_weights.keys() if not re.match('fid.*', k)}
    torch.save(weights_without_fid, output_weights_path)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="the path of the checkpoint from which the weights will be extracted",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="the output path of the extracted weights",
    )
    return parser.parse_args()

def main():
    args = get_args()
    extract_from_checkpoint(args.checkpoint_path, args.out_path)

if __name__=="__main__":
    main()