from __future__ import print_function
import os, shutil
from configs_bfi import get_main_parser
args = get_main_parser(mode='train').parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.device_ids[1:-1]
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torchvision.utils as vutils
import time
import numpy as np
import lpips
import sys
from torch.autograd import grad
from model import DGWGAN, Inversion
from torch.utils.tensorboard import SummaryWriter
import metrics_utils
from easydict import EasyDict
from torchvision.transforms import ToPILImage
from data import CelebA, FaceScrub, FFHQ, Vggface, LFW, Vggface2
torch.set_printoptions(sci_mode=False)
# -------------







# parser --------
from loguru import logger

os.makedirs(args.outdir, exist_ok=True)
# init logger --------
timeFormat = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
log_format = "<green>{time:MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
logger.remove()
logger.add(sys.stderr, format=log_format)
LOGGER_PATH = f"./{args.outdir}/train_bfi-{timeFormat}.log"
logger.add(LOGGER_PATH)
print = logger.info
print("==== logger init finish ====")

print("================================")
print(args)
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

FEATURE_TYPE_LIST = ['arcface', 'facenet', 'baidu', 'adaface', 'retinaface']
CUR_FEATURE_TYPE = next((substr for substr in FEATURE_TYPE_LIST if substr in args.outdir), None)
args.nz = 128 if CUR_FEATURE_TYPE == 'facenet' else 512


# GOLBALS
from FaceAttr.FaceAttrSolver import FaceAttrSolver
from facenet_pytorch.models.mtcnn import MTCNN
from loss.idloss.idenLossSolver import IdenLoss
from loss.parsing_loss.parsingLossSolver import ParseLoss
writer_counter = 0
time_interval = int(time.time())
loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
attr_model = FaceAttrSolver(epoches=100,batch_size=32,learning_rate=1e-2,model_type='Resnet18',optim_type='SGD',momentum=0.9,pretrained=True,loss_type='BCE_loss',exp_version='v7',device=device)
# facialFeature_model = MTCNN(margin=0, thresholds=[0.65, 0.75, 0.75], device=device)
# iden_loss = IdenLoss(device)
# parse_loss = ParseLoss(device)

from featureExtractZoo import featureExtractZoo
extractor = featureExtractZoo('iresnet100', args, True, None)


def freeze(net):
    for p in net.parameters():
        p.requires_grad_(False)
def unfreeze(net):
    for p in net.parameters():
        p.requires_grad_(True)
def gradient_penalty(x, y,DG):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape).cuda()
    z = x + alpha * (y - x)
    z = z.detach().cuda()
    z.requires_grad = True

    o = DG(z)
    g = grad(o, z, grad_outputs = torch.ones(o.size()).cuda(), create_graph = True)[0].view(z.size(0), -1)
    gp = ((g.norm(p = 2, dim = 1) - 1) ** 2).mean()

    return gp

def channelExpand(img):
    return img.clone().detach().repeat(1,3,1,1)


def calFeatureLoss(recon, img):
    labels = torch.tensor([0] * len(recon))
    recon_feature, img_feature, LABELS = extractor.extract(recon, img, labels, 0, None, True)
    return F.mse_loss(torch.tensor(recon_feature), torch.tensor(img_feature))


def cal_loss(recon, img, loss3, reduction='mean'): # ANCHOR cal_loss
    loss_dict = EasyDict()
    loss = torch.tensor(0,device=device,dtype=torch.float32)

    loss_dict.mse = F.mse_loss(recon, img,reduction=reduction)
    loss_dict.lpips = loss_fn_vgg(recon, img).mean() if reduction == 'mean' else loss_fn_vgg(recon, img).sum()
    loss_dict.gan = loss3.mean() if reduction == 'mean' else loss3.sum()
    loss_dict.attr = nn.L1Loss()(attr_model.predict(recon), attr_model.predict(img))
    # loss_dict.idenloss = iden_loss(recon, img)
    loss_dict.feature = calFeatureLoss(recon, img)
    # loss_dict.parseloss = parse_loss(recon, img)

    loss += loss_dict.mse
    loss += loss_dict.lpips
    loss += loss_dict.gan
    # loss += loss_dict.idenloss
    # loss += loss_dict.parseloss
    loss += loss_dict.attr*10
    loss += loss_dict.feature * 100

    loss /= len(loss_dict)
    return loss, loss_dict



#NOTE - train
def train(DG, INV, log_interval, device, data_loader, inv_optimizer,dg_optimizer, epoch,img_size):
    global time_interval, writer_counter

    INV.train()
    DG.train()

    for batch_idx, (imgs, feature, _) in enumerate(data_loader):
        imgs,feature = imgs.to(device), feature.to(device)
        # b, c, h, w
        freeze(INV)
        unfreeze(DG)
        dg_optimizer.zero_grad()

        '''训练 DG - 真实图像'''
        batch_size = imgs.size(0)
        label = torch.full((batch_size,), 1, dtype=imgs.dtype, device=device)
        real_prob = nn.Sigmoid()(DG(imgs))
        loss3_real = nn.BCELoss()(real_prob, label)
        loss3_real.backward()

        '''训练 DG - 虚假图像'''
        recon = INV(feature).to(device).detach()

        label.fill_(0)
        fake_prob = nn.Sigmoid()(DG(recon))
        loss3_fake = nn.BCELoss()(fake_prob,label)
        loss3_fake.backward()

        dg_optimizer.step()

        '''训练 INV'''
        freeze(DG)
        unfreeze(INV)
        inv_optimizer.zero_grad()

        recon = INV(feature)
        label.fill_(1)
        fake_prob = nn.Sigmoid()(DG(recon))
        loss3_g = nn.BCELoss()(fake_prob,label)

        loss, loss_dict = cal_loss(recon, imgs.float(), loss3_g)

        loss.backward()
        inv_optimizer.step()

        # output
        if batch_idx % log_interval == 0:
            print(f'\nTrain Epoch: {epoch} [{batch_idx * len(feature)}/{len(data_loader.dataset)}]\n\tloss3_real: {loss3_real:.6f}\n\tloss3_fake: {loss3_fake:.6f}\n\tCost: {int(time.time())-time_interval}')
            time_interval = int(time.time())
            for x in loss_dict:
                print(f'\t{x}: {loss_dict[x]:6f}')

            # loss可视化
            writer.add_scalar("loss3_real",loss3_real, writer_counter)
            writer.add_scalar("loss3_fake",loss3_fake, writer_counter)
            for x in loss_dict:
                writer.add_scalar(x,loss_dict[x], writer_counter)
            writer_counter+=1
        # 输出图像进行观察
        if batch_idx % (log_interval*40) == 0:
            imgs = imgs.float()
            truth = imgs[0:data_loader.batch_size]
            inverse = recon[0:data_loader.batch_size]
            alternating_images = [img for pair in zip(inverse, truth) for img in pair]
            alternating_tensor = torch.stack(alternating_images)
            grid = vutils.make_grid(alternating_tensor, nrow=inverse.size(0)//4, padding=2, normalize=True, scale_each=False)
            vutils.save_image(grid, f"{args.outdir}/train_{writer_counter}.png", normalize=False)



# ANCHOR validation
def test(DG, INV, device, data_loader, epoch, msg):
    global time_interval
    time_interval = int(time.time())
    num_counter = 0

    # INV.eval()
    DG.eval()

    loss_dict_total = EasyDict()
    loss_dict_total.attr = 0
    loss_dict_total.mse = 0

    with torch.no_grad():
        plot = True
        for (imgs,feature,_) in data_loader:
            imgs,feature = imgs.to(device),feature.to(device)
            batch_size = imgs.size(0)
            label = torch.full((batch_size,), 1, dtype=imgs.dtype, device=device)

            recon = INV(feature)
            fake_prob = nn.Sigmoid()(DG(recon))
            label.fill_(1)
            loss3_g = nn.BCELoss()(fake_prob,label)

            loss, loss_dict = cal_loss(recon, imgs.float(), loss3_g)

            for x in loss_dict:
                if x not in loss_dict_total:
                    loss_dict_total[x] = 0
                loss_dict_total[x] += loss_dict[x]
            num_counter += 1

            if plot:
                imgs = imgs.float()
                truth = imgs[0:64]
                inverse = recon[0:64]
                alternating_images = [img for pair in zip(inverse, truth) for img in pair]
                alternating_tensor = torch.stack(alternating_images)
                grid = vutils.make_grid(alternating_tensor, nrow=inverse.size(0)//4, padding=2, normalize=True, scale_each=False)
                vutils.save_image(grid, '{}/recon_{}_{}.png'.format(args.outdir,msg.replace(" ", ""), epoch), normalize=False)
                #图像写入tensorboard
                writer.add_image("recon"+msg, grid, epoch)
                plot = False

    for x in loss_dict_total:
        loss_dict_total[x] /= num_counter

    loss = 0
    for x in loss_dict_total:
        loss+=loss_dict_total[x]
    loss/= len(loss_dict_total.keys())

    # loss可视化
    for x in loss_dict:
        writer.add_scalar(x+"_test",loss_dict[x], epoch)

    print(f'\t[{msg}] Cost: {int(time.time())-time_interval}')
    for x in loss_dict_total:
        print(f'\t{x}: {loss_dict_total[x]:4f}')
    print(f'\tloss: {loss:4f}')


    time_interval = int(time.time())
    return loss, loss_dict_total, loss_dict_total.attr, loss_dict_total.mse

def metric(INV,device,data_loader, epoch):
    import tqdm
    from metricZoo import MetricZoo

    # INV.eval()
    num_counter = 0
    progress_bar = tqdm.tqdm(total=len(data_loader), desc="metric", leave=True)
    total_dict = EasyDict()
    metricZoo = MetricZoo(device,logger)

    with torch.no_grad():
        for batch_idx,(imgs,feature,_) in enumerate(data_loader):
            # REVIEW
            # imgs = F.interpolate(imgs, size=(args.img_size,args.img_size), mode='bilinear', align_corners=True)
            imgs,feature = imgs.to(device), feature.to(device)
            recon = INV(feature).to(device).detach()
            # vutils.save_image(torch.cat((recon,imgs)), './tmp.png', normalize=False)
            batch_dict = metricZoo.cal_img_metric(imgs, recon)
            # input("continue?")
            for x in batch_dict:
                if x not in total_dict:
                    total_dict[x] = 0
                total_dict[x] += batch_dict[x]
            # 把batch的指标显示在进度条上
            showstr = ""
            for x in batch_dict:
                showstr += f"{x}: {batch_dict[x]:4f} | "

            print(showstr) if batch_idx%100==0 else None  # 为的是logger临时记录一下batch
            progress_bar.set_description(showstr)
            progress_bar.update(1)
            num_counter += 1

    fid = metricZoo.cal_fid_metric(None, None, False, True)
    is_mean, is_std = metricZoo.cal_is_metric(None, False, True)

    for x in total_dict:
        total_dict[x] /= num_counter

    total_dict.fid = fid
    total_dict.is_mean = is_mean
    total_dict.is_std = is_std

    print(f"\n metric result:")
    for x in total_dict:
        writer.add_scalar(f"metric/{x}", total_dict[x], epoch)
        print(f"\t{x}: {total_dict[x]:4f}")
    print(f"{args.bfi_pretrain}")
    return None

def save_model(epoch,inversion,inv_optimizer,best_recon_loss,loss_name,msg):
    if loss_name != "epoch":
        state = {
            'epoch': epoch,
            'model': inversion.state_dict(),
            # 'optimizer': inv_optimizer.state_dict(),
            loss_name: best_recon_loss
        }
    else:
        state = {
            'epoch': epoch,
            'model': inversion.state_dict(),
            # 'optimizer': inv_optimizer.state_dict(),
            'loss': best_recon_loss
        }
    torch.save(state, f'./{args.outdir}/inversion_{loss_name}.pth')
    print("模型参数已经更新: "+loss_name+msg)

# ANCHOR main
def main():
    # 打印cal_loss的源代码,做校验
    import inspect
    print(inspect.getsource(cal_loss))
    if not any(atype in args.metric_data_path and atype in args.data_path and atype in args.imgdir for atype in ["adaalign", "nbalign"]):
        raise Exception(f"数据集路径不对,请检查:{args.metric_data_path} {args.data_path} {args.imgdir}")

    # 加载dataset
    imgTransform = transforms.Compose([
        transforms.Resize((args.img_size,args.img_size)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5,),(0.5,)),
    ])
    if 'celeba' in args.outdir:
        train_set = CelebA(args.data_path, transform=imgTransform,datanum=0)
        test1_set = CelebA(args.data_path, transform=imgTransform, datanum=1)
        test2_set = CelebA(args.data_path, transform=imgTransform,datanum=2)
        print("[*] Using Celeba dataset...")
    elif 'facescrub' in args.outdir:
        train_set = CelebA(args.data_path, transform=imgTransform,datanum=3)
        test1_set = FaceScrub('./dataset', transform=imgTransform, train=True)
        test2_set = FaceScrub('./dataset', transform=imgTransform, train=False)
        print("[*] Using Facescrub dataset...")
    elif 'ffhq' in args.outdir:
        train_set = FFHQ(args.data_path, transform=imgTransform)
        test1_set = train_set
        test2_set = train_set
        print("[*] Using FFHQ dataset...")
    elif 'vggface2' in args.outdir:
        train_set = Vggface2(args.data_path, args.imgdir, transform=imgTransform)
        test1_set = train_set
        test2_set = train_set
    elif 'vggface' in args.outdir:
        train_set = Vggface(args.data_path, transform=imgTransform)
        test1_set = train_set
        test2_set = train_set

    metric_set = LFW(args.metric_data_path, transform=imgTransform)

    # 加载data_loader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    # test1_loader = torch.utils.data.DataLoader(test1_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    # test2_loader = torch.utils.data.DataLoader(test2_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    metric_set = torch.utils.data.DataLoader(metric_set, batch_size=args.batch_size, shuffle=False, **kwargs)

    # 加载model
    device_list = list(range(len(eval(args.device_ids))))
    INV = Inversion(nc=args.nc, ngf=args.ngf, nz=args.nz, truncation=args.truncation, c=args.c).to(device)
    INV = nn.DataParallel(INV,device_ids= device_list,)
    DG = DGWGAN(in_dim=args.nc,dim=args.img_size).to(device)
    DG = nn.DataParallel(DG,device_ids= device_list,)

    # 加载优化器
    inv_optimizer = optim.Adam(INV.parameters(), lr=args.lr, betas=(0.5, 0.999), amsgrad=True)
    dg_optimizer = optim.Adam(DG.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # 允许加载反演器 TODO
    isload = False
    if isload:
        path = f''
        checkpoint = torch.load(path,map_location=device)
        INV.load_state_dict(checkpoint['model'])
        INV = INV.to(device)
        epoch = checkpoint['epoch']
        print("=> loaded inversion checkpoint '{}' (epoch {}".format(path, epoch))
        resumeEpoch = epoch

    # Train inversion model
    best_attr_loss = 999
    best_mse_loss = 999
    start_epoch = resumeEpoch if isload else 0
    print(f"[*] Start from epoch{start_epoch}")
    for epoch in range(start_epoch, args.epochs + 1):
        time_interval = int(time.time())
        train(DG, INV, args.log_interval, device, train_loader, inv_optimizer,dg_optimizer,epoch,args.img_size)
        print(f"\ttrain cost: {int(time.time())-time_interval}")

        if 'ffhq' in args.outdir or 'vggface' in args.outdir:
            mse_loss = best_mse_loss
            attr_loss = best_attr_loss
            print(f"[*] Skip test")
        else:
            _, _, attr_loss,mse_loss  = test(DG, INV, device, test1_loader, epoch, 'test1') #重建loss:确认生成网络的输出图像与真实图像的差异
            test(DG, INV, device, test2_loader, epoch, 'test2')

        if epoch % 1 == 0:
            save_model(epoch,INV,inv_optimizer,(best_attr_loss+best_mse_loss)/2,f"epoch{epoch}",f"_epoch{epoch}")
            metric(INV, device, metric_set, epoch)

        # 保存模型参数
        if mse_loss < best_mse_loss:
            best_mse_loss = mse_loss
            save_model(epoch,INV,inv_optimizer,best_mse_loss,"mse",f"_epoch{epoch}")
            shutil.copyfile('./{}/recon_test1_{}.png'.format(args.outdir,epoch), f'./{args.outdir}/best_mse_test1.png') #文件拷贝
            shutil.copyfile('./{}/recon_test2_{}.png'.format(args.outdir,epoch), f'./{args.outdir}/best_mse_test2.png')
        if attr_loss < best_attr_loss:
            best_attr_loss = attr_loss
            save_model(epoch,INV,inv_optimizer,best_attr_loss,"attr",f"_epoch{epoch}")
            shutil.copyfile('./{}/recon_test1_{}.png'.format(args.outdir,epoch), f'./{args.outdir}/best_attr_test1.png') #文件拷贝
            shutil.copyfile('./{}/recon_test2_{}.png'.format(args.outdir,epoch), f'./{args.outdir}/best_attr_test2.png')



if __name__ == '__main__':
    main()
    NEWLOGGER_PATH = f"./{args.outdir}/[FINISH]train_bfi-{timeFormat}.log"
    shutil.move(LOGGER_PATH, NEWLOGGER_PATH)
