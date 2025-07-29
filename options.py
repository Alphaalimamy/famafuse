import torch
import argparse

parser = argparse.ArgumentParser(description='options')
BASE = "D://RESEARCH/IMAGE FUSION/FUSION DATASET/SPECT-MRI"

device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser.add_argument('--DEVICE', type=str, default=device)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--patch_size', type=int, default=256)
parser.add_argument('--seed', type=int, default=3407)

parser.add_argument('--dir_train', type=str, default=f'{BASE}/train/')  # CT PET SPECT
parser.add_argument('--dir_test', type=str, default=F'{BASE}/test/')  # CT PET SPECT


parser.add_argument('--img_type1', type=str, default='SPECT/')  # CT PET SPECT
parser.add_argument('--img_type2', type=str, default='MRI/')

parser.add_argument('--model_save_path', type=str, default='./modelsave/SPECT/')  # CT PET SPECT
parser.add_argument('--model_save_name', type=str, default='Famafuse.pth')
parser.add_argument('--temp_dir', type=str, default='./temp/SPECT-MRI')  # CT PET SPECT
parser.add_argument('--img_save_dir', type=str, default='result/SPECT-MRI')  # CT PET SPECT

args = parser.parse_args()