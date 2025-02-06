from __future__ import print_function
import os, shutil
import argparse
from configs_bfi import get_main_parser
args = get_main_parser(mode='train').parse_args()
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = args.device_ids[1:-1]
# os.environ['MKL_NUM_THREADS'] = '1'
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
from model_blocks import DGWGAN, Inversion7
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
device = torch.device(f"cuda:{int(args.device_ids[1])}" if use_cuda else "cpu")
print(torch.cuda.is_available())
print(f"using device: {device}")
kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}
writer = SummaryWriter(f'{args.outdir}/runs')

torch.manual_seed(args.seed)

# -------------
from featureExtractZoo import featureExtractZoo

DATASET_TYPE = 'val'
CONTINUE = False

FEATURE_TYPE_LIST = ['arcface', 'facenet', 'baidu', 'adaface', 'retinaface','iresnet100']
IMG_TYPE_LIST = ['celeba', 'facescrub','lfw','vggface','ffhq']
CUR_FEATURE_TYPE = next((substr for substr in FEATURE_TYPE_LIST if substr in args.data_path), None)  # 控制模型结构的feature类型， 不一定等于attack的特征类型
CUR_IMG_TYPE = next((substr for substr in IMG_TYPE_LIST if substr in args.data_path), None) # datapath控制，
if args.lfwType != 0: CUR_IMG_TYPE += str(args.lfwType)
# TARGET_FEATURE_TYPE = next((substr for substr in FEATURE_TYPE_LIST if substr in args.target_feature_type), None)  # attack的特征类型
TARGET_FEATURE_TYPE = args.target_feature_type
npz_epoch = int(''.join(re.findall(r'\d', args.bfi_pretrain.split('/')[-1])))
NPZPATH = args.outdir + f"/reconFeature_{TARGET_FEATURE_TYPE}_{CUR_IMG_TYPE}_{DATASET_TYPE}_{npz_epoch}.npz"

extractor = featureExtractZoo(TARGET_FEATURE_TYPE, args, CONTINUE, NPZPATH,device)
args.nz = 128 if CUR_FEATURE_TYPE == 'facenet' else 512

def freeze(net):
    for p in net.parameters():
        p.requires_grad_(False)
def unfreeze(net):
    for p in net.parameters():
        p.requires_grad_(True)


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
            # print(imgs)
            if num_counter<=finished_idx:
                print(f"skip batch_{num_counter}")
                num_counter+=1
                progress_bar.update(1)
                continue

            featureOri = feature[1].squeeze().to(device)
            feature = feature[0].squeeze().to(device)
            # print(imgs)
            if 'dcgan' in args.bfi_pretrain:
                feature = featureOri
            # print(feature.shape)
            recon = INV(feature).to(device).detach()
            # print(recon)
            # if 'dcgan' in args.bfi_pretrain:
            #     import sys
            #     sys.path.append("/home/gank/NBNet/src")
            #     from util.util import prewhiten
            #     recon = np.clip((recon.cpu().numpy()+1.0)/2.0, 0, 1)
            #     recon = [np.expand_dims(prewhiten(x),axis=0) for x in recon]
            #     recon = np.concatenate(recon,axis=0)
            #     recon=torch.tensor(recon).to(device).detach()
            # print(recon)
            # print(recon.shape)
            vutils.save_image(recon, f"{args.outdir}/testrecon.png", normalize=False)
            # 原始特征彩图 vs 反演特征灰图
            if CUR_FEATURE_TYPE == TARGET_FEATURE_TYPE: # 提取同类型特征，不需提取ori
                curLen = extractor.extract(recon, imgs, _, num_counter, featureOri)
            else:
                curLen = extractor.extract(recon, imgs, _, num_counter)

            # curLen = extractor.extract(recon, imgs, _, num_counter)

            progress_bar.set_description(f"extracted {curLen}")
            progress_bar.update(1)
            num_counter += 1

            if batch_idx % (args.log_interval) == 0:
                print(f"batch_idx: {batch_idx}, condition: {batch_idx % (args.log_interval)}")

                imgs = imgs.float().to(device)
                truth = imgs[0:args.batch_size]
                inverse = recon[0:args.batch_size]
                alternating_images = [img for pair in zip(inverse, truth) for img in pair]
                alternating_tensor = torch.stack(alternating_images)
                grid = vutils.make_grid(alternating_tensor, nrow=inverse.size(0) // 4, padding=2, normalize=False,
                                        scale_each=False)
                vutils.save_image(grid, f"{args.outdir}/test_{batch_idx}.png", normalize=False)

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
    # 加载model
    # device_list = list(range(len(eval(args.device_ids))))
    device_list=eval(args.device_ids)
    print(device_list)
    if 'bfi' in args.bfi_pretrain:
        INV = Inversion(nc=args.nc, ngf=args.ngf, nz=args.nz, truncation=args.truncation, c=args.c).to(device)
    elif 'nbnet' in args.bfi_pretrain:
        INV = NbNet(nl=32,nz=128).to(device)
    INV = nn.DataParallel(INV,device_ids= device_list,)
    DG = DGWGAN(in_dim=args.nc,dim=args.img_size).to(device)
    DG = nn.DataParallel(DG,device_ids= device_list,)

    # 允许加载反演器
    isload = True
    if isload:
        checkpoint = torch.load(args.bfi_pretrain,map_location=device)
        INV.load_state_dict(checkpoint['model'])
        INV = INV.to(device)
        epoch = checkpoint['epoch']
        print("=> loaded inversion checkpoint '{}' (epoch {}".format(args.bfi_pretrain, epoch))

        extract(INV,device,data_loader)


if __name__ == '__main__':
    main()
    print(f"==== extract finish ====")
    NEWLOGGER_PATH = f"./{args.outdir}/[FINISH]extract_feature-{timeFormat}.log"
    shutil.move(LOGGER_PATH, NEWLOGGER_PATH)
