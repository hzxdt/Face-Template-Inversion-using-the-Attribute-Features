from cgitb import reset
import torch
import numpy as np
from tqdm import *
from skimage.metrics import structural_similarity as compare_ssim
from pytorch_msssim import ms_ssim
# from pytorch_fid.inception import InceptionV3
# from torchvision.models.inception import inception_v3
import metrics_utils
import torch.nn.functional as F
from FaceAttr.FaceAttrSolver import FaceAttrSolver
from facenet_pytorch.models.mtcnn import MTCNN
from torchvision.transforms import ToPILImage
from loss.fiveloss.facial_feature_utils import get_eyes, get_nose, get_mouth
from easydict import EasyDict
# import cupy as cp
# from torch_sqrtm import sqrtm, torch_matmul_to_array, np_to_gpu_tensor
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from torch.nn.functional import adaptive_avg_pool2d
from functools import partial
from scipy.stats import entropy
import torch.nn as nn
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

print = None

class MetricZoo:
    def __init__(self,device, logger):
        global print
        print = logger.info
        
        self.device = device
        self.attr_model = FaceAttrSolver(epoches=100,batch_size=32,learning_rate=1e-2,model_type='Resnet18',optim_type='SGD',momentum=0.9,pretrained=True,loss_type='BCE_loss',exp_version='v7',device=device)
        self.facialFeature_model = MTCNN(margin=0, thresholds=[0.65, 0.75, 0.75], device=device)
        
        # self.fid_model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device) # dims: 2048
        self.fid_model = FrechetInceptionDistance()
        # self.is_model = inception_v3(pretrained=True, transform_input=False).type(torch.cuda.FloatTensor)
        self.is_model = InceptionScore()
        self.fid_model = self.fid_model.to(device)
        self.is_model = self.is_model.to(device)
        
        self.fid_pred_recon = []
        self.fid_pred_ori = []
        self.is_pred = []
        
        self.is_model.eval()
    
    # group metric ----------------------------------------------------------------------------------------------------------
    def cal_img_metric(self, ori: torch.Tensor, recon: torch.Tensor):
        rtn = EasyDict()
        rtn.ssim = self.ssim(ori, recon)
        rtn.msssim = self.msssim(ori, recon)
        rtn.mse = self.mse(ori, recon)
        rtn.sfa = self.sfa(ori, recon)
        rtn.five = self.five(ori, recon)
        self.cal_fid_metric(recon, ori, False, False)
        self.cal_is_metric(recon, False, False)
        return rtn
    

    # img metric ----------------------------------------------------------------------------------------------------------
    
    def ssim(self, original: torch.Tensor, reconstructed: torch.Tensor):
        original_images = metrics_utils.tensor_to_image(original.detach().cpu())
        reconstructed_images = metrics_utils.tensor_to_image(reconstructed.detach().cpu())
        ssim_score = 0
        for i in range(original_images.shape[0]):
            # ssim_score += compare_ssim(original_images[i, :, :, :], reconstructed_images[i, :, :, :], channel_axis=2)
            ssim_score += compare_ssim(original_images[i, :, :, :].squeeze(), reconstructed_images[i, :, :, :].squeeze(), multichannel=True, channel_axis=2)
        return ssim_score / original_images.shape[0]

    def msssim(self, original: torch.Tensor, reconstructed: torch.Tensor):
        return ms_ssim(original, reconstructed, data_range=1, size_average=False, win_size=3).mean()

    def mse(self, reconstructed: torch.Tensor, original: torch.Tensor):
        return F.mse_loss(reconstructed, original).item()

    def sfa(self, original: torch.Tensor, reconstructed: torch.Tensor):
        original = self.to_three_channels(original)
        reconstructed = self.to_three_channels(reconstructed)
        return nn.L1Loss()(self.attr_model.predict(reconstructed), self.attr_model.predict(original))
    
    def five(self, img: torch.Tensor, recon: torch.Tensor):
        five_loss = 0
        useful_pixel = 0
        eyes_list, noses_list, mouths_list, eyes_masks_list, noses_masks_list, mouths_masks_list = self.get_facial_feature(img)
        cropped_eyes_list, cropped_noses_list, cropped_mouths_list = self.crop_facial_feature(recon, eyes_masks_list, noses_masks_list, mouths_masks_list)

        # 计算损失
        for original, cropped in zip([eyes_list, noses_list, mouths_list], [cropped_eyes_list, cropped_noses_list, cropped_mouths_list]):
            five_loss += F.mse_loss(original, cropped, reduction='sum')

        # 计算有用像素
        for mask_list in [eyes_masks_list, noses_masks_list, mouths_masks_list]:
            for mask in mask_list:
                useful_pixel += np.count_nonzero(~mask)
        five_loss /= useful_pixel
        five_loss /= 3
        return five_loss
    
    # fid metric ----------------------------------------------------------------------------------------------------------
    def cal_fid_metric(self, recon, ori, isRestart, isFinal):
        if isRestart:
            self.fid_model.reset()
        if isFinal:
            value = self.fid_model.compute()
            return value
        else:   # 输入要求：(B, C, H, W)，uint8，299*299
            recon = (F.interpolate(recon, 299) * 255).to(torch.uint8)
            ori = (F.interpolate(ori, 299) * 255).to(torch.uint8)
            self.fid_model.update(recon, real=False)
            self.fid_model.update(ori, real=True)
            return None
        
            
    def compute_fid_pred(self, x):  # x.shape: (B,1,H,W)
        with torch.no_grad():
            pred = self.fid_model(x)[0]
        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        return pred
    
    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, device, eps=1e-6):

        array_to_tensor = partial(np_to_gpu_tensor, device)    
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = sqrtm(torch_matmul_to_array(array_to_tensor(sigma1), array_to_tensor(sigma2)), array_to_tensor, disp=False)

        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = sqrtm(torch_matmul_to_array(array_to_tensor(sigma1 + offset), array_to_tensor(sigma2 + offset)), array_to_tensor)

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        diff_ = array_to_tensor(diff)
        return (torch_matmul_to_array(diff_, diff_) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
            
    # inception score metric ----------------------------------------------------------------------------------------------------------
    def cal_is_metric(self, recon, isRestart, isFinal):
        '''
        return: mean, std
        '''
        if isRestart:
            self.is_model.reset()
        if isFinal:
            value = self.is_model.compute()
            return value
        else:
            recon = (F.interpolate(recon, 299) * 255).to(torch.uint8)
            self.is_model.update(recon)
            return None
    
    # feature metric ----------------------------------------------------------------------------------------------------------
    
    def cal_feature_metric(self, recon,ori,label,PATH,thre): # RANK#ALL
        '''

        tar_normal_t_ori_LIST / a1 vs a2,a3
        tar_type1_t_recon_LIST / a1' vs a1
        tar_type2_t_recon_LIST / a1' vs a1 a2 a3 
        tar_normal_t_recon_LIST / a1' vs a2a3
        '''
        
        recon = recon.cuda().squeeze()
        ori = ori.cuda().squeeze()
        dict_name = [np.where(label == i)[0] for i in np.unique(label)] #装有每个类对应的索引

        far_t_ori_LIST = []
        tar_normal_t_ori_LIST = []
        tar_type1_t_recon_LIST = []
        tar_type2_t_recon_LIST = []
        tar_normal_t_recon_LIST = []

        table_csim_all = self.getTableCsim(recon,ori,dict_name,0)# .cuda()
        table_csim_exself =self.getTableCsim(recon,ori,dict_name,1)# .cuda()
        table_csim_real_exself = self.getTableCsim(ori,ori,dict_name,1)# .cuda()
        table_csim_real_exclz = self.getTableCsim(ori,ori,dict_name,2)# .cuda()

        csim_var = table_csim_all.diagonal().var()
        cossim_total = table_csim_all.diagonal().mean() # 计算table_csim对角线总和

        for clz in dict_name:
            # table_csim_all
            clz_table_csim = table_csim_all[clz][:,clz]
            clz_fla = clz_table_csim.flatten()
            tar_type2_t_recon_LIST += clz_fla
            
            # table_csim_real_exself
            clz_table_csim = table_csim_real_exself[clz][:,clz]
            clz_fla = clz_table_csim.flatten()
            clz_fla = clz_fla[clz_fla!=-float('inf')]
            tar_normal_t_ori_LIST += clz_fla
            
            # table_csim_exself
            clz_table_csim = table_csim_exself[clz][:,clz]
            clz_fla = clz_table_csim.flatten()
            clz_fla = clz_fla[clz_fla!=-float('inf')]
            tar_normal_t_recon_LIST += clz_fla
            
        def getRsl(t,farlist,*metricList):
            rtn = []
            for ml in metricList:
                if len(ml) == 0:
                    rtn.append(-999)
                else:
                    rtn.append((ml>t).sum().item() / len(ml))
            return rtn
        
        far_t_ori_LIST = table_csim_real_exclz.flatten()
        far_t_ori_LIST = far_t_ori_LIST[far_t_ori_LIST!=-float('inf')]
        tar_type1_t_recon_LIST = table_csim_all.diagonal()
        
        # far_t_ori_LIST = far_t_ori_LIST.cpu()
        # tar_normal_t_ori_LIST = tar_normal_t_ori_LIST.cpu()
        # tar_type1_t_recon_LIST = tar_type1_t_recon_LIST.cpu()
        # tar_type2_t_recon_LIST = tar_type2_t_recon_LIST.cpu()
        # tar_normal_t_recon_LIST = tar_normal_t_recon_LIST.cpu()

        t_0 = np.percentile(far_t_ori_LIST, (1-0.1)*100) if thre == None else thre[0]   # 10% FAR
        rsl_1 = getRsl(t_0,far_t_ori_LIST,
                        tar_normal_t_ori_LIST,
                        tar_normal_t_recon_LIST,
                        tar_type1_t_recon_LIST,
                        tar_type2_t_recon_LIST)
        t_1 = np.percentile(far_t_ori_LIST, (1-0.01)*100) if thre == None else thre[1]  # 1% FAR
        rsl_2 = getRsl(t_1,far_t_ori_LIST,
                        tar_normal_t_ori_LIST,
                        tar_normal_t_recon_LIST,
                        tar_type1_t_recon_LIST,
                        tar_type2_t_recon_LIST)
        t_2 = np.percentile(far_t_ori_LIST, (1-0.001)*100) if thre == None else thre[2] # 0.1% FAR
        rsl_3 = getRsl(t_2,far_t_ori_LIST,
                        tar_normal_t_ori_LIST,
                        tar_normal_t_recon_LIST,
                        tar_type1_t_recon_LIST,
                        tar_type2_t_recon_LIST)
        t_3 = np.percentile(far_t_ori_LIST, (1-0.0001)*100) if thre == None else thre[3] # 0.01% FAR
        rsl_4 = getRsl(t_3,far_t_ori_LIST,
                        tar_normal_t_ori_LIST,
                        tar_normal_t_recon_LIST,
                        tar_type1_t_recon_LIST,
                        tar_type2_t_recon_LIST)
        
        print(f"csim: {cossim_total}")
        print(f"t: {t_0}\t{t_1}\t{t_2}\t{t_3}")
        print(f"tar_normal_t_ori_LIST: {rsl_1[0]}\t{rsl_2[0]}\t{rsl_3[0]}\t{rsl_4[0]}")
        print(f"tar_normal_t_recon_LIST: {rsl_1[1]}\t{rsl_2[1]}\t{rsl_3[1]}\t{rsl_4[1]}")
        print(f"tar_type1_t_recon_LIST: {rsl_1[2]}\t{rsl_2[2]}\t{rsl_3[2]}\t{rsl_4[2]}")
        print(f"tar_type2_t_recon_LIST: {rsl_1[3]}\t{rsl_2[3]}\t{rsl_3[3]}\t{rsl_4[3]}")

        print(PATH+'\n')
        
        print('='*120)
        
        return (t_0,t_1)
    
    # roc metric ----------------------------------------------------------------------------------------------------------
    
    def get_roc_pic(self, feature_type,dataset_type, dataset):
    
        FAR_type1 = []
        FAR_type2 = []
        TAR_type1 = []
        TAR_type2 = []
        ROC_AUC_type1 = []
        ROC_AUC_type2 = []
        
        if dataset_type == 'ffhq':
            rsl_type1, rsl_type2, threlist = self.get_roc_line(*self.load_origin_data(dataset,dataset_type));                                                                                     FAR_type1.append(rsl_type1[0]);FAR_type2.append(rsl_type2[0]);TAR_type1.append(rsl_type1[1]);TAR_type2.append(rsl_type2[1]);ROC_AUC_type1.append(rsl_type1[2]);ROC_AUC_type2.append(rsl_type2[2])
            rsl_type1, rsl_type2, _ = self.get_roc_line(*self.load_data(f'/root/autodl-tmp/BFISR/out_ffhq_sr2only/recon_feature_baidu_val.npz'), dataset_type, threlist);                                  FAR_type1.append(rsl_type1[0]);FAR_type2.append(rsl_type2[0]);TAR_type1.append(rsl_type1[1]);TAR_type2.append(rsl_type2[1]);ROC_AUC_type1.append(rsl_type1[2]);ROC_AUC_type2.append(rsl_type2[2])
            rsl_type1, rsl_type2, _ = self.get_roc_line(*self.load_data(f'/root/autodl-tmp/BFISR/out_ffhq_loss125_sr2only/recon_feature_baidu_val.npz'), dataset_type, threlist);                                  FAR_type1.append(rsl_type1[0]);FAR_type2.append(rsl_type2[0]);TAR_type1.append(rsl_type1[1]);TAR_type2.append(rsl_type2[1]);ROC_AUC_type1.append(rsl_type1[2]);ROC_AUC_type2.append(rsl_type2[2])
            rsl_type1, rsl_type2, _ = self.get_roc_line(*self.load_data(f'/root/autodl-tmp/BFISR/out_ffhq_loss1235_sr2only/recon_feature_baidu_val.npz'), dataset_type, threlist);                                  FAR_type1.append(rsl_type1[0]);FAR_type2.append(rsl_type2[0]);TAR_type1.append(rsl_type1[1]);TAR_type2.append(rsl_type2[1]);ROC_AUC_type1.append(rsl_type1[2]);ROC_AUC_type2.append(rsl_type2[2])
            rsl_type1, rsl_type2, _ = self.get_roc_line(*self.load_data(f'/root/autodl-tmp/BFISR/out_ffhq_sr2only/recon_feature_nbnet_baidu_val.npz'), dataset_type, threlist);                                  FAR_type1.append(rsl_type1[0]);FAR_type2.append(rsl_type2[0]);TAR_type1.append(rsl_type1[1]);TAR_type2.append(rsl_type2[1]);ROC_AUC_type1.append(rsl_type1[2]);ROC_AUC_type2.append(rsl_type2[2])
            rsl_type1, rsl_type2, _ = self.get_roc_line(*self.load_data(f'/root/autodl-tmp/BFISR/out_ffhq_sr2only/recon_feature_gastylegan_celeba_baidu.npz'), dataset_type, threlist);                                  FAR_type1.append(rsl_type1[0]);FAR_type2.append(rsl_type2[0]);TAR_type1.append(rsl_type1[1]);TAR_type2.append(rsl_type2[1]);ROC_AUC_type1.append(rsl_type1[2]);ROC_AUC_type2.append(rsl_type2[2])
        if dataset_type == 'lfw':
            rsl_type1, rsl_type2, threlist = self.get_roc_line(*self.load_origin_data(dataset,dataset_type));                                                                                     FAR_type1.append(rsl_type1[0]);FAR_type2.append(rsl_type2[0]);TAR_type1.append(rsl_type1[1]);TAR_type2.append(rsl_type2[1]);ROC_AUC_type1.append(rsl_type1[2]);ROC_AUC_type2.append(rsl_type2[2])
            rsl_type1, rsl_type2, _ = self.get_roc_line(*self.load_data(f'/root/autodl-tmp/BFISR/out_ffhq_sr2only/recon_feature_lfw_baidu_val.npz'), dataset_type, threlist);                                  FAR_type1.append(rsl_type1[0]);FAR_type2.append(rsl_type2[0]);TAR_type1.append(rsl_type1[1]);TAR_type2.append(rsl_type2[1]);ROC_AUC_type1.append(rsl_type1[2]);ROC_AUC_type2.append(rsl_type2[2])
            rsl_type1, rsl_type2, _ = self.get_roc_line(*self.load_data(f'/root/autodl-tmp/BFISR/out_ffhq_loss125_sr2only/recon_feature_lfw_baidu_val.npz'), dataset_type, threlist);                                  FAR_type1.append(rsl_type1[0]);FAR_type2.append(rsl_type2[0]);TAR_type1.append(rsl_type1[1]);TAR_type2.append(rsl_type2[1]);ROC_AUC_type1.append(rsl_type1[2]);ROC_AUC_type2.append(rsl_type2[2])
            rsl_type1, rsl_type2, _ = self.get_roc_line(*self.load_data(f'/root/autodl-tmp/BFISR/out_ffhq_loss1235_sr2only/recon_feature_lfw_baidu_val.npz'), dataset_type, threlist);                                  FAR_type1.append(rsl_type1[0]);FAR_type2.append(rsl_type2[0]);TAR_type1.append(rsl_type1[1]);TAR_type2.append(rsl_type2[1]);ROC_AUC_type1.append(rsl_type1[2]);ROC_AUC_type2.append(rsl_type2[2])
            rsl_type1, rsl_type2, _ = self.get_roc_line(*self.load_data(f'/root/autodl-tmp/BFISR/out_ffhq_nbnet_percept/recon_feature_baidu_val_lfw.npz'), dataset_type, threlist);                                  FAR_type1.append(rsl_type1[0]);FAR_type2.append(rsl_type2[0]);TAR_type1.append(rsl_type1[1]);TAR_type2.append(rsl_type2[1]);ROC_AUC_type1.append(rsl_type1[2]);ROC_AUC_type2.append(rsl_type2[2])
            rsl_type1, rsl_type2, _ = self.get_roc_line(*self.load_data(f'/root/autodl-tmp/BFISR/out_ffhq_sr2only/recon_feature_gastylegan_lfw_baidu.npz'), dataset_type, threlist);                                  FAR_type1.append(rsl_type1[0]);FAR_type2.append(rsl_type2[0]);TAR_type1.append(rsl_type1[1]);TAR_type2.append(rsl_type2[1]);ROC_AUC_type1.append(rsl_type1[2]);ROC_AUC_type2.append(rsl_type2[2])
        if dataset_type == 'celebahq':
            rsl_type1, rsl_type2, threlist = self.get_roc_line(*self.load_origin_data(dataset,dataset_type));                                                                                     FAR_type1.append(rsl_type1[0]);FAR_type2.append(rsl_type2[0]);TAR_type1.append(rsl_type1[1]);TAR_type2.append(rsl_type2[1]);ROC_AUC_type1.append(rsl_type1[2]);ROC_AUC_type2.append(rsl_type2[2])
            rsl_type1, rsl_type2, _ = self.get_roc_line(*self.load_data(f'/root/autodl-tmp/BFISR2/BFISR/out_celebAHQ/recon_feature_baidu_val.npz'), dataset_type, threlist);                                  FAR_type1.append(rsl_type1[0]);FAR_type2.append(rsl_type2[0]);TAR_type1.append(rsl_type1[1]);TAR_type2.append(rsl_type2[1]);ROC_AUC_type1.append(rsl_type1[2]);ROC_AUC_type2.append(rsl_type2[2])
            rsl_type1, rsl_type2, _ = self.get_roc_line(*self.load_data(f'/root/autodl-tmp/BFISR/out_ffhq_loss125_sr2only/recon_feature_lfw_baidu_val.npz'), dataset_type, threlist);                                  FAR_type1.append(rsl_type1[0]);FAR_type2.append(rsl_type2[0]);TAR_type1.append(rsl_type1[1]);TAR_type2.append(rsl_type2[1]);ROC_AUC_type1.append(rsl_type1[2]);ROC_AUC_type2.append(rsl_type2[2])
            rsl_type1, rsl_type2, _ = self.get_roc_line(*self.load_data(f'/root/autodl-tmp/BFISR/out_ffhq_loss1235_sr2only/recon_feature_lfw_baidu_val.npz'), dataset_type, threlist);                                  FAR_type1.append(rsl_type1[0]);FAR_type2.append(rsl_type2[0]);TAR_type1.append(rsl_type1[1]);TAR_type2.append(rsl_type2[1]);ROC_AUC_type1.append(rsl_type1[2]);ROC_AUC_type2.append(rsl_type2[2])
            rsl_type1, rsl_type2, _ = self.get_roc_line(*self.load_data(f'/root/autodl-tmp/BFISR2/BFISR/out_celebAHQ_nbnet_percept/recon_feature_baidu_val.npz'), dataset_type, threlist);                                  FAR_type1.append(rsl_type1[0]);FAR_type2.append(rsl_type2[0]);TAR_type1.append(rsl_type1[1]);TAR_type2.append(rsl_type2[1]);ROC_AUC_type1.append(rsl_type1[2]);ROC_AUC_type2.append(rsl_type2[2])
            rsl_type1, rsl_type2, _ = self.get_roc_line(*self.load_data(f'/root/autodl-tmp/BFISR/out_ffhq_sr2only/recon_feature_gastylegan_facescrub_baidu.npz'), dataset_type, threlist);                                  FAR_type1.append(rsl_type1[0]);FAR_type2.append(rsl_type2[0]);TAR_type1.append(rsl_type1[1]);TAR_type2.append(rsl_type2[1]);ROC_AUC_type1.append(rsl_type1[2]);ROC_AUC_type2.append(rsl_type2[2])
        colors = ['darkred','crimson','orange','gold','mediumseagreen','steelblue', 'mediumpurple'][:len(FAR_type1)]
        # names = ['Baseline','BFISG','Nbnet','GA-stylegan'][:len(FAR_type1)]
        # REVIEW
        names = ['Baseline','BFISG','BFISG-4loss','BFISG-3loss','Nbnet','GA-stylegan'][:len(FAR_type1)]

        plt.figure(figsize=(20, 20), dpi=100)
        plt.rcParams['font.size'] = 30
        for fpr,tpr,roc_auc,color,name in zip(FAR_type1,TAR_type1,ROC_AUC_type1,colors,names):
            plt.plot(fpr,tpr,color = color, lw = 5, label = f'{name} (AUC: {roc_auc:0.4f})')
            # plt.plot([0, 1], [0, 1], '--', lw=5, color = 'grey')
        plt.xlim([0,1]);plt.ylim([0,1]);plt.xlabel('FAR',fontsize=40);plt.ylabel('TAR',fontsize=40)
        plt.title('ROC',fontsize=60)
        plt.legend(loc = 'lower right',fontsize=27)
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.tick_params(axis='both', labelsize=30, pad=10)
        plt.savefig(f'./roc-{dataset_type}-{feature_type}-type1.png')
        
        if dataset_type == 'ffhq':
            return
        plt.figure(figsize=(20, 20), dpi=100)
        plt.rcParams['font.size'] = 30
        for fpr,tpr,roc_auc,color,name in zip(FAR_type2,TAR_type2,ROC_AUC_type2,colors,names):
            plt.plot(fpr,tpr,color = color, lw = 5, label = f'{name} roc area:({roc_auc:0.4f})')
            # plt.plot([0, 1], [0, 1], '--', lw=5, color = 'grey')
        plt.xlim([0,1]);plt.ylim([0,1]);plt.xlabel('FAR',fontsize=40);plt.ylabel('TAR',fontsize=40)
        plt.title('ROC',fontsize=60)
        plt.legend(loc = 'lower right',fontsize=27)
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.tick_params(axis='both', labelsize=30, pad=10)
        plt.savefig(f'./roc-{dataset_type}-{feature_type}-type2.png')
    
    def get_roc_line(self,recon,ori,label,dataset_type,threlist = None):
        '''
        far_t_ori_LIST / a1 vs b1b2b3
        tar_normal_t_ori_LIST / a1 vs a2,a3 / baseline
        tar_type1_t_recon_LIST / a1' vs a1 (type1)
        tar_normal_t_recon_LIST / a1' vs a2a3 (type2)
        '''
        recon = recon.cuda().squeeze()
        ori = ori.cuda().squeeze()
        dict_name = [np.where(label == i)[0] for i in np.unique(label)] #装有每个类对应的索引
        accvalue = 1000

        far_t_ori_LIST = []
        tar_normal_t_ori_LIST = []
        tar_type1_t_recon_LIST = []

        tar_normal_t_recon_LIST = []
        
        table_csim_all = self.getTableCsim(recon,ori,dict_name,0)
        table_csim_exself = self.getTableCsim(recon,ori,dict_name,1)
        # table_csim_real_exself = getTableCsim(ori,ori,dict_name,1)
        table_csim_real_exclz = self.getTableCsim(ori,ori,dict_name,2)

        for clz in dict_name:
            
            # table_csim_exself
            clz_table_csim = table_csim_exself[clz][:,clz]
            clz_fla = clz_table_csim.flatten()
            clz_fla = clz_fla[clz_fla!=-float('inf')]
            tar_normal_t_recon_LIST += clz_fla
            
        
        far_t_ori_LIST = table_csim_real_exclz.flatten()
        far_t_ori_LIST = far_t_ori_LIST[far_t_ori_LIST!=-float('inf')]
        tar_type1_t_recon_LIST = table_csim_all.diagonal()
        
        farlist = np.linspace(0,1,accvalue)
        tarlist_type1 = []
        tarlist_type2 = []
        tarlist_base = []
        thre = []
        
        tar_normal_t_recon_LIST = torch.tensor(tar_normal_t_recon_LIST).cuda()
        tar_type1_t_recon_LIST = torch.tensor(tar_type1_t_recon_LIST).cuda()
        len_tar_normal_t_recon_LIST = len(tar_normal_t_recon_LIST)
        len_tar_type1_t_recon_LIST = len(tar_type1_t_recon_LIST)
        
        if threlist == None:
            for i in tqdm(range(0,accvalue,1)):
                thre.append(cp.percentile(cp.asarray(far_t_ori_LIST), (1-i/accvalue)*100))
            # for i in tqdm(range(0,accvalue,1)):
            #     thre.append(np.percentile(np.asarray(far_t_ori_LIST), (1-i/accvalue)*100))
        else:
            thre = threlist
        for t in thre:
            tarlist_type1.append((tar_type1_t_recon_LIST>t.item()).sum().item() / len_tar_type1_t_recon_LIST)
            if dataset_type != 'ffhq':
                tarlist_type2.append((tar_normal_t_recon_LIST>t.item()).sum().item() / len_tar_normal_t_recon_LIST)
            else:
                tarlist_type2.append(0)
            
        
        auc_type1 = auc(farlist,tarlist_type1)
        auc_type2 = auc(farlist,tarlist_type2)
        
        
        return (farlist,tarlist_type1,auc_type1),(farlist,tarlist_type2,auc_type2),thre
    
    
    
    # common -----------------------------------------------------------------------------------------------------------
    
    def to_three_channels(self, img):
        if img.shape[1] == 1:
            return img.repeat(1, 3, 1, 1)
        return img
    
    def get_facial_feature(self, img):
        eyes_list = []
        noses_list = []
        mouths_list = []
        eyes_masks_list = []
        noses_masks_list = []
        mouths_masks_list = []

        for x in img:
            x_pil = ToPILImage()(x)
            x_numpy = np.array(x_pil)
            _, __, landmarks = self.facialFeature_model.detect(x_pil, landmarks=True)

            if landmarks is None:
                # 创建空的特征和掩码
                empty_feature = torch.zeros(x_numpy.shape[::-1]).to(self.device)
                empty_mask = np.zeros(x_numpy.shape[:2], dtype=bool)

                # 将空特征和掩码添加到对应的列表
                eyes_list.append(empty_feature)
                noses_list.append(empty_feature)
                mouths_list.append(empty_feature)
                eyes_masks_list.append(empty_mask)
                noses_masks_list.append(empty_mask)
                mouths_masks_list.append(empty_mask)
                continue

            landmarks = landmarks[0]
            landmarks = torch.tensor(landmarks.astype(float), dtype=torch.float32)
            landmarks = np.around(landmarks).numpy().astype(np.int16)

            # 提取特征和掩码
            eyes, eyes_mask = get_eyes(x_numpy, landmarks)
            nose, nose_mask = get_nose(x_numpy, landmarks)
            mouth, mouth_mask = get_mouth(x_numpy, landmarks)

            # 处理特征和掩码，然后添加到列表
            eyes = torch.as_tensor(eyes).to(self.device) / 255
            nose = torch.as_tensor(nose).to(self.device) / 255
            mouth = torch.as_tensor(mouth).to(self.device) / 255
            eyes = eyes.permute(2, 0, 1)
            nose = nose.permute(2, 0, 1)
            mouth = mouth.permute(2, 0, 1)

            eyes_list.append(eyes)
            noses_list.append(nose)
            mouths_list.append(mouth)
            eyes_masks_list.append(eyes_mask)
            noses_masks_list.append(nose_mask)
            mouths_masks_list.append(mouth_mask)
            
        eyes_list = torch.stack(eyes_list)
        noses_list = torch.stack(noses_list)
        mouths_list = torch.stack(mouths_list)

        return eyes_list, noses_list, mouths_list, eyes_masks_list, noses_masks_list, mouths_masks_list

                
    def crop_facial_feature(self, recon, eyes_masks, noses_masks, mouths_masks):
        cropped_eyes_list = []
        cropped_noses_list = []
        cropped_mouths_list = []

        for i in range(len(recon)):
            eyes_mask = eyes_masks[i]
            nose_mask = noses_masks[i]
            mouth_mask = mouths_masks[i]

            cropped_eyes = recon[i] * torch.as_tensor(~eyes_mask).to(self.device)
            cropped_noses = recon[i] * torch.as_tensor(~nose_mask).to(self.device)
            cropped_mouths = recon[i] * torch.as_tensor(~mouth_mask).to(self.device)

            cropped_eyes_list.append(cropped_eyes)
            cropped_noses_list.append(cropped_noses)
            cropped_mouths_list.append(cropped_mouths)
            
        cropped_eyes_list = torch.stack(cropped_eyes_list)
        cropped_noses_list = torch.stack(cropped_noses_list)
        cropped_mouths_list = torch.stack(cropped_mouths_list)

        return cropped_eyes_list, cropped_noses_list, cropped_mouths_list
    
    def getTableCsim(self,x,y,dict_name,type):
        if type == 0:
            table = torch.zeros((x.shape[0],y.shape[0]))
            for i in tqdm(range(x.shape[0])):
                x_i = x[i].repeat(y.shape[0],1)
                table[i] = torch.cosine_similarity(x_i, y.squeeze(), dim=1)
        if type == 1: #去掉对角线
            table = torch.zeros((x.shape[0],y.shape[0]))
            for i in tqdm(range(x.shape[0])):
                x_i = x[i].repeat(y.shape[0],1)
                table[i] = torch.cosine_similarity(x_i, y.squeeze(), dim=1)
            table[range(table.shape[0]),range(table.shape[0])] = -float('inf')
        if type == 2: #去掉同类
            table = torch.zeros((x.shape[0],y.shape[0]))
            for i in tqdm(range(x.shape[0])):
                x_i = x[i].repeat(y.shape[0],1)
                table[i] = torch.cosine_similarity(x_i, y.squeeze(), dim=1)
            for clz in dict_name:
                for x in clz:
                    table[x,clz] = -float('inf')
        return table
    def getTableL1(self,x,y,dict_name,type):
        if type == 0:
            table = torch.zeros((x.shape[0], y.shape[0]))
            for i in tqdm(range(x.shape[0])):
                x_i = x[i].unsqueeze(0)  # 将x[i]变成一个形状为(1, m)的张量，以便进行广播
                # 计算x_i与y中每个元素之间的L1距离
                # 这里我们使用了unsqueeze和广播机制来避免显式地重复x_i
                table[i] = torch.sum(torch.abs(x_i - y), dim=1)  # 在特征维度上计算绝对值差异的和
        if type == 1: #去掉对角线
            table = torch.zeros((x.shape[0],y.shape[0]))
            for i in tqdm(range(x.shape[0])):
                x_i = x[i].unsqueeze(0)
                table[i] = torch.sum(torch.abs(x_i - y), dim=1)
            table[range(table.shape[0]),range(table.shape[0])] = -float('inf')
        if type == 2: #去掉同类
            table = torch.zeros((x.shape[0],y.shape[0]))
            for i in tqdm(range(x.shape[0])):
                x_i = x[i].unsqueeze(0)
                table[i] = torch.sum(torch.abs(x_i - y), dim=1)
            for clz in dict_name:
                for x in clz:
                    table[x,clz] = -float('inf')
        return table
    def getTableL2(self,x,y,dict_name,type):
        if type == 0:
            table = torch.zeros((x.shape[0], y.shape[0]))
            for i in tqdm(range(x.shape[0])):
                x_i = x[i].unsqueeze(0)  # 将x[i]变成一个形状为(1, m)的张量，以便进行广播
                # 计算x_i与y中每个元素之间的L1距离
                # 这里我们使用了unsqueeze和广播机制来避免显式地重复x_i
                table[i] = torch.sum((x_i - y) ** 2, dim=1)  # 在特征维度上计算绝对值差异的和
        if type == 1: #去掉对角线
            table = torch.zeros((x.shape[0],y.shape[0]))
            for i in tqdm(range(x.shape[0])):
                x_i = x[i].unsqueeze(0)
                table[i] = torch.sum((x_i - y) ** 2, dim=1)
            table[range(table.shape[0]),range(table.shape[0])] = -float('inf')
        if type == 2: #去掉同类
            table = torch.zeros((x.shape[0],y.shape[0]))
            for i in tqdm(range(x.shape[0])):
                x_i = x[i].unsqueeze(0)
                table[i] = torch.sum((x_i - y) ** 2, dim=1)
            for clz in dict_name:
                for x in clz:
                    table[x,clz] = -float('inf')
        return table
    @staticmethod
    def load_data(self, PATH):
        recon = np.load(PATH, allow_pickle=True)
        recon_feature = recon['recon']/1.
        ori_feature = recon['ori']/1.
        label = recon['label']        
        
        print(f"recon_feature.shape: {recon_feature.shape[0]} {recon_feature.shape[1]}")
        print(f'ori_feature.shape: {ori_feature.shape}')
        print(f'label.shape: {label.shape}')
        return torch.tensor(recon_feature),torch.tensor(ori_feature),label
    
    @staticmethod
    def load_origin_data(self, ds, dataset_type):
        if dataset_type == 'ffhq':
            feature = ds.feature
            if not ds.isFeature255:
                feature = feature/255.0
            return torch.tensor(feature), torch.tensor(feature), list(range(len(feature))), dataset_type
        if dataset_type == 'lfw':
            if max(ds.features[0])>2:
                feature = ds.features/255.0
            return torch.tensor(feature), torch.tensor(feature), ds.labels, dataset_type
        if dataset_type == 'celebahq':
            if max(ds.features[0])>2:
                feature = ds.features/255.0
            return torch.tensor(feature), torch.tensor(feature), ds.labels, dataset_type


