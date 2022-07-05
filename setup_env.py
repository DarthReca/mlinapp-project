from modules import utils
from modules.utils import bcolors
import sys
import os
import os.path
import urllib.request
import pkg_resources
import argparse

errors = False
missing_packages = False
missing_dataset = True
missing_weights = True

parser = argparse.ArgumentParser()
parser.add_argument("-y", "--yes_all", dest="yes_all", action="store_true")
args = parser.parse_args()

######################################
print('== Checking Python version ==')

version = ".".join(map(str, sys.version_info[:3]))
if sys.version_info.major != 3 or sys.version_info.minor != 9:
    print(f"{bcolors.WARNING}Warning: the project was developed using Python 3.9, but you are using Python {version} consider switching to Python 3.9 if you encounter problems while running the scripts (like this one){bcolors.ENDC}")
else:
    print(f"{bcolors.SUCCESS}You are using Python {version} ✔{bcolors.ENDC}")


######################################
print('\n== Checking requirements ==')

installed_packages = [p.project_name for p in pkg_resources.working_set]

with open("requirements.txt") as reqfile:
    for line in reqfile.readlines():
        package = line.strip()
        if package not in installed_packages:
            print(
                f"{bcolors.WARNING}Package '{package}' is not installed{bcolors.ENDC}")
            missing_packages = True
        else:
            print(
                f"{bcolors.SUCCESS}Package '{package}' is already installed{bcolors.ENDC}")

if missing_packages:
    if args.yes_all or utils.query_yes_no("Some packages are missing.\nWould you like to run 'pip install -r requirements.txt' now?", default=None):
        result = os.system("pip install -r requirements.txt")
        if result == 0:
            print(
                f"{bcolors.SUCCESS}Requirements successfully installed ✔{bcolors.ENDC}")
            missing_packages = False
        else:
            print(
                f"{bcolors.ERROR}Something went wrong while installing requirements, please try again later ✘{bcolors.ENDC}")
            errors = True
else:
    print(f"{bcolors.SUCCESS}All required packages are already installed ✔{bcolors.ENDC}")


######################################
print('\n== Checking dataset ==')
if os.path.isfile("data/celeba/img_align_celeba.zip"):
    print(f"{bcolors.INFO}The dataset had already been downloaded{bcolors.ENDC}")
else:
    print(f"{bcolors.WARNING}The dataset hasn't been downloaded yet{bcolors.ENDC}")
    if args.yes_all or utils.query_yes_no("Would you like to download the dataset now?", default=None):
        try:
            urllib.request.urlretrieve("https://www.dropbox.com/s/ydfwka7plnrd7dz/img_align_celeba.zip?dl=1",
                                       "data/celeba/img_align_celeba.zip", utils.reporthook)
            print(f"{bcolors.INFO}Download completed{bcolors.ENDC}")
        except KeyboardInterrupt:
            print(f"{bcolors.WARNING}\nDownload cancelled, please delete the partial dataset before attempting again{bcolors.ENDC}")

if os.path.isfile("data/celeba/img_align_celeba.zip"):
    computed_md5 = utils.compute_md5("data/celeba/img_align_celeba.zip")
    correct_md5 = "00d2c5bc6d35e252742224ab0c1e8fcb"
    if computed_md5 != correct_md5:
        print(f"{bcolors.ERROR}The checksum of the dataset doesn't match the correct one! ✘\nPlease delete the dataset and try downloading it again{bcolors.ENDC}")
        errors = True
    else:
        print(
            f"{bcolors.SUCCESS}The dataset passed the integrity check ✔{bcolors.ENDC}")
        missing_dataset = False


######################################
print('\n== Checking pretrained weights ==')
if os.path.isfile("weights/pretrained.pth"):
    print(f"{bcolors.INFO}The pretrained weights had already been downloaded{bcolors.ENDC}")
else:
    print(f"{bcolors.WARNING}The pretrained weights haven't been downloaded yet{bcolors.ENDC}")
    if args.yes_all or utils.query_yes_no("Would you like to download the weights now?", default=None):
        try:
            urllib.request.urlretrieve(
                "https://www.dropbox.com/s/44g36acwkxlsllw/inject0_orig.pth?dl=1", "weights/pretrained.pth", utils.reporthook)
            print(f"{bcolors.INFO}Download completed{bcolors.ENDC}")
        except KeyboardInterrupt:
            print(f"{bcolors.WARNING}\nDownload cancelled, please delete the partial weights before attempting again{bcolors.ENDC}")

if os.path.isfile("weights/pretrained.pth"):
    computed_md5 = utils.compute_md5("weights/pretrained.pth")
    correct_md5 = "600a826a9df47d85f9f96357052e50bd"
    if computed_md5 != correct_md5:
        print(f"{bcolors.ERROR}The checksum of the weights doesn't match the correct one! ✘\nPlease delete those weights and try downloading them again{bcolors.ENDC}")
        errors = True
    else:
        print(
            f"{bcolors.SUCCESS}The weights passed the integrity check ✔{bcolors.ENDC}")
        missing_weights = False


######################################
print('\n== Setup ended ==')
if errors:
    print(f"{bcolors.ERROR}The setup finished with errors ✘{bcolors.ENDC}")
else:
    if any([missing_packages, missing_dataset, missing_weights]):
        print(
            f"{bcolors.WARNING}The setup finished without errors, but you are missing:")
        if missing_packages:
            print("• some packages")
        if missing_dataset:
            print("• the dataset")
        if missing_weights:
            print("• the pretrained weights")
        print(f"Consider running this script again{bcolors.ENDC}")
    else:
        print(
            f"{bcolors.SUCCESS}Everything went smoothly, you are ready to get training ✔{bcolors.ENDC}")
