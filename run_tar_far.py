# 批处理
import subprocess
from configs_bfi import get_main_parser
import re
import glob
args = get_main_parser(mode='train').parse_args()
import logging
import numpy as np
import torch
from metricZoo import MetricZoo
import io
import scipy.io as sio
import base64
import requests
import math
np.set_printoptions(precision=4)
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(f"metric_result/{args.outdir}", "metric result of tar")

def run_single(args, epoch):
    command = [
        'python', 'extract_feature.py',
        '--bfi_pretrain', f"./{args.outdir}/inversion_epoch{epoch}.pth",
        '--outdir', args.outdir,
        '--data-path', f"{args.metric_data_path}",
        '--target_feature_type', args.target_feature_type,  # REVIEW 记得修改
        '--device-ids', "[2]",   # gai
        # "--lfwType", ""   # 是否是500
    ]
    print(f"=> running {command}")
    result = subprocess.run(command, capture_output=True, text=True)
    print(f"=> finish {command}")
    return result

def load_data(metric_data_path):
    recon = np.load(metric_data_path, allow_pickle=True)
    recon_feature = recon['recon']/1.
    ori_feature = recon['ori']/1.
    label = recon['label']

    print(f'recon_feature.shape: {recon_feature.shape}')
    print(f'ori_feature.shape: {ori_feature.shape}')
    print(f'label.shape: {label.shape}')
    return torch.tensor(recon_feature),torch.tensor(ori_feature),label

def getCsim(metric_data_path, epoch):
    recon, ori, label = load_data(metric_data_path)
    recon = recon.cuda().squeeze()
    ori = ori.cuda().squeeze()
    dict_name = [np.where(label == i)[0] for i in np.unique(label)] #装有每个类对应的索引

    score = MetricZoo.getTableCsim(None, recon,ori,dict_name,0)# .cuda()
    # table_csim_exself =MetricZoo.getTableCsim(recon,ori,dict_name,1)# .cuda()
    # table_csim_real_exself = MetricZoo.getTableCsim(ori,ori,dict_name,1)# .cuda()
    # table_csim_real_exclz = MetricZoo.getTableCsim(ori,ori,dict_name,2)# .cuda()
    csim = score.diagonal().mean()

    return csim


def sendNpzData(npzPath, epoch):
    api_url = 'http://58c95b2c.r21.cpolar.top/evalroc'
    npz_base64 = base64.b64encode(open(npzPath,'rb').read()).decode()
    print(f"Sending {npzPath}")
    data = {'npz_content': npz_base64, 'epoch': epoch}
    response = requests.post(api_url, json=data)
    if response.status_code == 200:
        result = response.json()
        result = base64.b64decode(result['result_base64'])
        result_mat = sio.loadmat(io.BytesIO(result))
        veriFAR = result_mat['veriFAR']
        type1VR = result_mat['type1VR']
        type1Thresholds = result_mat['type1Thresholds']
        type2VR = result_mat['type2VR']
        type2Thresholds = result_mat['type2Thresholds']

        # print("VR:")
        # print(VR)
        # print("veriFAR:")
        # print(veriFAR)
        # print("thresholds:")
        # print(thresholds)
        print("请求成功")
        return {"veriFAR": veriFAR, "type1VR": type1VR, "type1Thresholds": type1Thresholds, "type2VR": type2VR, "type2Thresholds": type2Thresholds}
    else:
        print("请求失败，状态码：", response.status_code)
        return None


def recordMetricData(mat_data, csim, epoch):
    type1VR = mat_data['type1VR']
    type1Thresholds = mat_data['type1Thresholds']
    type2VR = mat_data['type2VR']
    type2Thresholds = mat_data['type2Thresholds']
    veriFAR = mat_data['veriFAR']
    idxs = [37, 46, 55, 64]               # 0.0001, 0.001, 0.01, 0.1
    selected_far = [0.0001, 0.001, 0.01, 0.1]
    selected_type1vr = type1VR[0][idxs]
    selected_type1threshold = type1Thresholds[0][idxs]
    selected_type2vr = type2VR[0][idxs]
    selected_type2threshold = type2Thresholds[0][idxs]

    print(f"epoch {epoch} csim: {csim}")
    for thre, vr, far in zip(selected_type1threshold, selected_type1vr, selected_far):
        print(f"epoch {epoch} FAR_{far}/type1VR: {vr}")
        print(f"epoch {epoch} FAR_{far}/type1Threshold: {thre}")
        writer.add_scalar(f'FAR_{far}/type1VR', vr, epoch)
        writer.add_scalar(f'FAR_{far}/type1Threshold', thre, epoch)
    for thre, vr, far in zip(selected_type2threshold, selected_type2vr, selected_far):
        print(f"epoch {epoch} FAR_{far}/type2VR: {vr}")
        print(f"epoch {epoch} FAR_{far}/type2Threshold: {thre}")
        writer.add_scalar(f'FAR_{far}/type2VR', vr, epoch)
        writer.add_scalar(f'FAR_{far}/type2Threshold', thre, epoch)

def getDoneInversion():
    completed_epochs = []
    for file in glob.glob(f"./{args.outdir}/inversion_epoch*.pth"):
        match = re.search(r'inversion_epoch(\d+)\.pth', file)
        if match:
            epoch = int(match.group(1))
            completed_epochs.append(epoch)
    return completed_epochs

def getDoneNpz():
    completed_epochs = []
    for file in glob.glob(f"./{args.outdir}/reconFeature*.npz"):
        match = re.search(r'reconFeature_(\w+)_(\w+)_(\w+)_(\d+)\.npz', file)
        if match:
            epoch = int(match.group(4))
            completed_epochs.append(epoch)
    return completed_epochs

def getReconFeaturePath(epoch):
    FEATURE_TYPE_LIST = ['arcface', 'facenet', 'baidu', 'adaface', 'retinaface','iresnet100']
    IMG_TYPE_LIST = ['celeba', 'facescrub','lfw','vggface','ffhq']
    DATASET_TYPE = 'val'
    CUR_IMG_TYPE = next((substr for substr in IMG_TYPE_LIST if substr in args.metric_data_path), None)
    TARGET_FEATURE_TYPE = next((substr for substr in FEATURE_TYPE_LIST if substr in args.target_feature_type), None)
    return args.outdir + f"/reconFeature_{TARGET_FEATURE_TYPE}_{CUR_IMG_TYPE}_{DATASET_TYPE}_{epoch}.npz"





completed_epochs = []
choice = input("0: 批量extract_feature.py\n1: 批量Metric BLUFR\n")
for file in sorted(glob.glob(f"./{args.outdir}/inversion_epoch*.pth"), key=lambda x: int(re.search(r'inversion_epoch(\d+)\.pth', x).group(1))):
    match = re.search(r'inversion_epoch(\d+)\.pth', file)
    if match:
        epoch = int(match.group(1))
        # featurePath = f"./{args.outdir}/reconFeature_{args.target_feature_type}_*.npz"
        if epoch not in list(range(200, 300+1)):
            print(f"skip epoch {epoch}")
            continue

        if epoch not in completed_epochs:
            if choice == '0':
                # 批量extract_feature.py
                print(f"=> running epoch {epoch}")
                rsl = run_single(args, epoch)
                if rsl.returncode == 0:
                    completed_epochs.append(epoch)
                    print(f"=> finish epoch {epoch}")
                else:
                    print(f"=> ERROR epoch {epoch}")
                    print(rsl.stderr)
                    pass
            elif choice == '1':
                # 批量Metric BLUFR
                print(f"=> running epoch {epoch} Metric BLUFR")

                csim = getCsim(getReconFeaturePath(epoch), epoch)
                record_data = sendNpzData(getReconFeaturePath(epoch), epoch)
                if record_data is not None:
                    recordMetricData(record_data, csim, epoch)
                    print(f"=> finish epoch {epoch} Metric BLUFR")
                else:
                    print("=> ERROR")

        else:
            print(f"epoch {epoch} is already completed, skip it")
