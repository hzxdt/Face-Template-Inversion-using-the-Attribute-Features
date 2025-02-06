from __future__ import print_function
import os, shutil
from configs_bfi import get_main_parser
args = get_main_parser(mode='train').parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.device_ids[1:-1]
os.environ['MKL_NUM_THREADS'] = '1'
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torchvision.utils as vutils
import time
import tqdm
import numpy as np
import sys
from torch.autograd import grad
# from model_blocks import DGWGAN, Inversion7
from model import Inversion
from torch.utils.tensorboard import SummaryWriter
import metrics_utils
from easydict import EasyDict
from torchvision.transforms import ToPILImage
from data import CelebA, FaceScrub, FFHQ, LFW, LFW500, Vggface, FFHQ500
from nbnet import NbNet
import re
# -------------







# parser --------
from loguru import logger

os.makedirs(args.outdir, exist_ok=True)
os.makedirs(f"{args.outdir}/recon_feature_dataset", exist_ok=True)
# init logger --------
timeFormat = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
log_format = "<green>{time:MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
logger.remove()
logger.add(sys.stderr, format=log_format)
LOGGER_PATH = f"./{args.outdir}/extract_feature-{timeFormat}.log"
logger.add(LOGGER_PATH)
print = logger.info
print("==== logger init finish ====")

print("================================")
[print(f'{arg}: {value}') for arg, value in sorted(vars(args).items())]
print("================================")


print(f"device: {torch.cuda.device_count()}")
if args.use_cudnn_benchmark:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

use_cuda = args.use_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}
writer = SummaryWriter(f'{args.outdir}/runs')

torch.manual_seed(args.seed)

# -------------
from featureExtractZoo import featureExtractZoo

DATASET_TYPE = 'val'
CONTINUE = True

FEATURE_TYPE_LIST = ['arcface', 'facenet', 'baidu', 'adaface', 'retinaface','iresnet100']
IMG_TYPE_LIST = ['celeba', 'facescrub','lfw','vggface','ffhq']
CUR_FEATURE_TYPE = next((substr for substr in FEATURE_TYPE_LIST if substr in args.bfi_pretrain), None)  # 控制模型结构的feature类型， 不一定等于attack的特征类型
CUR_IMG_TYPE = next((substr for substr in IMG_TYPE_LIST if substr in args.data_path), None) # datapath控制，
if args.lfwType != 0: CUR_IMG_TYPE += str(args.lfwType)
TARGET_FEATURE_TYPE = next((substr for substr in FEATURE_TYPE_LIST if substr in args.target_feature_type), None)  # attack的特征类型
npz_epoch = int(''.join(re.findall(r'\d', args.bfi_pretrain.split('/')[-1])))
ALIGN_TYPE = "adaalign" if "adaalign" in args.metric_data_path else "nbalign"
NPZPATH = args.outdir + f"/reconFeature_{TARGET_FEATURE_TYPE}_{CUR_IMG_TYPE}_{DATASET_TYPE}_{ALIGN_TYPE}.npz"

extractor = featureExtractZoo(TARGET_FEATURE_TYPE, args, CONTINUE, NPZPATH)
args.nz = 128 if CUR_FEATURE_TYPE == 'facenet' else 512



def extract(INV,device,data_loader):
    global time_interval, CONTINUE
    # # INV.eval()
    num_counter = 0
    progress_bar = tqdm.tqdm(total=len(data_loader), desc="extract", leave=True)

    finished_idx = -1
    if CONTINUE:
        try:
            finished_idx = np.load(NPZPATH)['batch_idx']
        except FileNotFoundError:
            print(f"Not found, creating an empty npz in {NPZPATH}")
            print(f"CONTINUE switch to False")
            CONTINUE = False
            np.savez(NPZPATH, recon=np.array([]),ori=np.array([]),label=np.array([]), batch_idx = -1)


    with torch.no_grad():
        for batch_idx,(imgs,feature,_) in enumerate(data_loader):
            if num_counter<=finished_idx:
                print(f"skip batch_{num_counter}")
                num_counter+=1
                progress_bar.update(1)
                continue

            featureOri = feature[1].squeeze().to(device)
            feature = feature[0].squeeze().to(device)
            if 'dcgan' in args.bfi_pretrain:
                feature = featureOri

            # recon = INV(feature).to(device).detach()
            # vutils.save_image(recon, f"{args.outdir}/testrecon.png", normalize=False)
            # 原始特征彩图 vs 反演特征灰图
            # if CUR_FEATURE_TYPE == FEATURE_TYPE: # 提取同类型特征，不需提取ori
            #     curLen = extractor.extract(recon, imgs, _, num_counter, featureOri)
            # else:
            #     curLen = extractor.extract(recon, imgs, _, num_counter)

            curLen = extractor.extract(imgs, imgs, _, num_counter)

            progress_bar.set_description(f"extracted {curLen}")
            progress_bar.update(1)
            num_counter += 1

    print(f"{NPZPATH}")
    return None

# ANCHOR main
def main():
    # check
    if not any(atype in args.metric_data_path and atype in args.data_path and atype in args.imgdir for atype in ["adaalign", "nbalign"]):
        raise Exception(f"数据集路径不对,请检查:{args.metric_data_path} {args.data_path} {args.imgdir}")
    # 加载dataset
    imgTransform = transforms.Compose([
        transforms.Resize((args.img_size,args.img_size)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5,),(0.5,)),
    ])
    if 'celeba' in CUR_IMG_TYPE:
        if DATASET_TYPE == 'train':
            test_set = CelebA(args.data_path, transform=imgTransform,datanum=0, isOriginalFeature=True) # train
        elif DATASET_TYPE == 'val':
            test_set = CelebA(args.data_path, transform=imgTransform,datanum=2, isOriginalFeature=True)
        print("[*] Using Celeba dataset...")
    elif 'facescrub' in CUR_IMG_TYPE:
        test_set = FaceScrub('./dataset', transform=imgTransform, isOriginalFeature=True, useAll=True)
        print("[*] Using FaceScrub dataset...")
    elif 'lfw500' in CUR_IMG_TYPE:
        test_set = LFW500(args.data_path, transform=imgTransform, isOriginalFeature=True)
    elif 'lfw' in CUR_IMG_TYPE:
        test_set = LFW(args.data_path, transform=imgTransform, isOriginalFeature=True)
    elif 'vggface' in CUR_IMG_TYPE:
        test_set = Vggface(args.data_path, transform=imgTransform, isOriginalFeature=True)
        print("[*] Using Vggface dataset...")
    elif 'ffhq' in CUR_IMG_TYPE:
        if 'ffhq14w' in args.bfi_pretrain:
            test_set = FFHQ500(args.data_path, transform=imgTransform, isOriginalFeature=True)
            print("[*] Using FFHQ500 dataset...")
        else:
            test_set = FFHQ(args.data_path, transform=imgTransform)
            print("[*] Using FFHQ dataset...")
    # 加载data_loader
    data_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)


    extract(None,device,data_loader)


if __name__ == '__main__':
    main()
    print(f"==== extract finish ====")
    NEWLOGGER_PATH = f"./{args.outdir}/[FINISH]extract_feature-{timeFormat}.log"
    shutil.move(LOGGER_PATH, NEWLOGGER_PATH)
