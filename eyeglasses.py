import argparse
import torch
import torchvision.transforms as T
from modules.neural_net.attgan_parts import Generator
from PIL import Image
import re

# retrieve arguments
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", type=str, required=True, help="The path of the image upon which eyeglasses will be added")
parser.add_argument("-a", "--attrs", type=str, required=True, choices=[ "custom", "damiano", "giulia", "daniele", "patrizio"], help="Which attributes should be used as target during generation, if 'custom' you have to modify the values inside the script before running it")
parser.add_argument("-w", "--weights", type=str, required=True, default="weights/pretrained_plus_dg2.pth", help="The path of the weights to be used during inference")
args = parser.parse_args()

# load the image
source_img = Image.open(args.path)

# preprocess the image
transform = T.Compose(
    [
        T.CenterCrop(min(source_img.size)),
        T.Resize(128),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)
preprocessed_source_img = transform(source_img)
preprocessed_source_img = torch.unsqueeze(preprocessed_source_img, dim=0)   # add dummy batch size

# instantiate the generator
generator = Generator()

# load the weights
state_dict = torch.load(args.weights, map_location="cpu")
generator_state_dict = {k[len("generator."):]:state_dict[k] for k in state_dict.keys() if re.match('generator.*', k)}
generator.load_state_dict(generator_state_dict)

# set target attributes
if args.attrs == "custom":
    print("If this is your first time using the 'custom' attributes, consider setting the desired ones by editing the code of this script")
    # modify the 0s and 1s here
    attrs = list({
        "Bald":0,
        "Bangs":0,
        "Black_Hair":0,
        "Blond_Hair":0,
        "Brown_Hair":0,
        "Bushy_Eyebrows":0,
        "Eyeglasses":0,
        "Male":0,
        "Mouth_Slightly_Open":0,
        "Mustache":0,
        "No_Beard":0,
        "Pale_Skin":0,
        "Young":0
    }.values())
elif args.attrs == "damiano":
    attrs = [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1]
elif args.attrs == "giulia":
    attrs = [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1]
elif args.attrs == "daniele":
    attrs = [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1]
elif args.attrs == "patrizio":
    attrs = [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1]

print("Target attributes", attrs)

# perform inference
generator.eval()
target_attrs_ts = torch.tensor(attrs, dtype=torch.float32)
target_attrs_ts[6] = 1  # force eyeglasses to 1
target_attrs_ts = (target_attrs_ts * 2 - 1) * 0.5  # shift to -0.5,0.5
target_attrs_ts = torch.unsqueeze(target_attrs_ts, dim=0)   # add dummy batch size

target_img_ts = generator(preprocessed_source_img, target_attrs_ts)[0]

def images_from_tensor(img: torch.Tensor) -> torch.Tensor:
    img = img.clone().detach()
    img = img.clamp_(-1, 1).sub_(-1).div(2)
    return (img * 255).round().byte()

target_img_ts = images_from_tensor(target_img_ts)

# show results
to_pil = T.ToPILImage()
target_img = to_pil(target_img_ts)
target_img.show()
