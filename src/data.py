from __future__ import print_function, division
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import copy
import torch


class FaceScrub(Dataset):
    def __init__(self, root, transform=None, target_transform=None, seed=666, train=True, useAll=False,
                 isOriginalFeature=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.isOriginalFeature = isOriginalFeature

        # 加载原始图像
        imgs = []
        input_path = os.path.join(self.root, 'img_facescrub.npy')
        imgs = np.load(input_path, allow_pickle=True)

        # 加载特征向量
        features = []
        input_path = os.path.join(self.root, 'feature_facescrub.npy')
        features = np.load(input_path, allow_pickle=True)
        featuresOriginal = copy.deepcopy(features)
        v_min = features.min(axis=0)
        v_max = features.max(axis=0)
        features = (features - v_min) / (v_max - v_min)

        # 加载name
        names = []
        with open('./dataset/output_img_facescrub.txt') as f:
            lines = f.readlines()
            for line in lines:
                name = line.split('\t')[-1].split('/')[2]
                names.append(name)
        self.num_cls = len(set(names))

        # 打乱数据集
        np.random.seed(seed)
        perm = np.arange(len(imgs))
        np.random.shuffle(perm)

        imgs = imgs[perm]
        names = np.array(names)[perm]
        features = features[perm]
        featuresOriginal = featuresOriginal[perm]

        # ori_feature = ori_feature[perm]

        if useAll == False:
            if train:  # 如果要作为训练集, 就取前80%
                self.features = features[0:int(0.8 * len(features))]
                self.featuresOriginal = featuresOriginal[0:int(0.8 * len(features))]
                self.imgs = imgs[0:int(0.8 * len(imgs))]
                self.names = names[0:int(0.8 * len(imgs))]
                self.name_set = sorted(set(self.names))
                # self.num_cls = len(self.names)
                # self.ori_feature = ori_feature[0:int(0.8 * len(data))]
            else:  # 如果不是训练集, 取后20%
                self.features = features[int(0.8 * len(imgs)):]
                self.featuresOriginal = featuresOriginal[int(0.8 * len(imgs)):]
                self.imgs = imgs[int(0.8 * len(imgs)):]
                self.names = names[int(0.8 * len(imgs)):]
                self.name_set = sorted(set(self.names))
                # self.num_cls = len(self.names)

                # self.ori_feature = ori_feature[int(0.8 * len(data)):]
        else:
            self.features = features
            self.featuresOriginal = featuresOriginal
            self.imgs = imgs
            self.names = names
            self.num_cls = len(set(self.names))
            self.name_set = sorted(set(self.names))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img, feature, name = self.imgs[index], self.features[index], self.name_set.index(self.names[index])
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.isOriginalFeature:
            return img, (feature, self.featuresOriginal[index]), name
        else:
            return img, feature, name


class CelebA_baidu(Dataset):
    def __init__(self, root, seed=666, transform=None, target_transform=None, datanum=0):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        # 加载特征向量
        input_path = os.path.join(self.root, 'celebA_total.npz')
        input = np.load(input_path, allow_pickle=True)
        labels = input['LABEL']
        images = input['IMG']
        # inter_labels = input['INTER_LABEL'] 真实标签
        features = input['FEATURE']

        v_min = images.min(axis=0)
        v_max = images.max(axis=0)
        images = (images - v_min) / (v_max - v_min)
        v_min = features.min(axis=0)
        v_max = features.max(axis=0)
        features = (features - v_min) / (v_max - v_min)

        target_classnum = 2167
        unique_labels, counts = np.unique(labels, return_counts=True)
        if datanum == 0:  # train
            order = np.argsort(-counts)[target_classnum:]
        elif datanum == 1 or datanum == 2:  # test
            order = np.argsort(-counts)[:target_classnum]
        elif datanum == 3:
            order = np.argsort(-counts)
        idx = []
        for i in order:
            idx.append(np.where(labels == unique_labels[i])[0])
        idx = np.concatenate(idx, axis=0)

        labels = labels[idx]

        unique_labels = np.unique(labels)
        for i, x in enumerate(unique_labels.tolist()):
            labels[labels == x] = i

        images = images[idx]
        # inter_labels = inter_labels[idx]
        features = features[idx]

        self.imgs = images
        self.features = features
        self.labels = labels

        # 作为测试集的话就打乱
        if datanum == 1 or datanum == 2:
            print(f"Checking celeba dataset...\n{labels}")
            np.random.seed(seed)
            perm = np.arange(len(images))
            np.random.shuffle(perm)
            images = images[perm]
            features = features[perm]
            labels = labels[perm]

        if datanum == 1:
            self.imgs = images[0:int(0.8 * len(images))]
            self.features = features[0:int(0.8 * len(images))]
            self.labels = labels[0:int(0.8 * len(images))]
            print(f"Loading celeba TRAIN dataset...\n{self.labels}")
        elif datanum == 2:
            self.imgs = images[int(0.8 * len(images)):]
            self.features = features[int(0.8 * len(images)):]
            self.labels = labels[int(0.8 * len(images)):]
            print(f"Loading celeba TEST dataset...\n{self.labels}")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img, feature, label = self.imgs[index], self.features[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img, feature, label


class CelebA(Dataset):
    def __init__(self, input_path, seed=666, transform=None, target_transform=None, datanum=0, isOriginalFeature=False):
        assert 'celeba' in input_path
        self.transform = transform
        self.target_transform = target_transform
        self.isOriginalFeature = isOriginalFeature

        # 加载特征向量
        input = np.load(input_path, allow_pickle=True)
        labels = input['label']
        images = input['img']
        features = input['feature']

        featuresOriginal = copy.deepcopy(features)

        v_min = features.min(axis=0)
        v_max = features.max(axis=0)
        features = (features - v_min) / (v_max - v_min)

        if 'arcface' in input_path:
            target_classnum = 2326
        elif 'facenet' in input_path:
            target_classnum = 2360
        elif 'baidu' in input_path:
            target_classnum = 2167
        unique_labels, counts = np.unique(labels, return_counts=True)
        if datanum == 0:  # train
            order = np.argsort(-counts)[target_classnum:]
        elif datanum == 1 or datanum == 2:  # test
            order = np.argsort(-counts)[:target_classnum]
        elif datanum == 3:
            order = np.argsort(-counts)
        idx = []
        for i in order:
            idx.append(np.where(labels == unique_labels[i])[0])
        idx = np.concatenate(idx, axis=0)

        labels = labels[idx]

        unique_labels = np.unique(labels)
        for i, x in enumerate(unique_labels.tolist()):
            labels[labels == x] = i

        images = images[idx]
        # inter_labels = inter_labels[idx]
        features = features[idx]
        featuresOriginal = featuresOriginal[idx]

        self.imgs = images
        self.features = features
        self.labels = labels
        self.featuresOriginal = featuresOriginal

        # 作为测试集的话就打乱
        if datanum == 1 or datanum == 2:
            print(f"Checking celeba dataset...\n{labels}")
            np.random.seed(seed)
            perm = np.arange(len(images))
            np.random.shuffle(perm)
            images = images[perm]
            features = features[perm]
            featuresOriginal = featuresOriginal[perm]
            labels = labels[perm]

        if datanum == 1:
            self.imgs = images[0:int(0.8 * len(images))]
            self.features = features[0:int(0.8 * len(images))]
            self.featuresOriginal = featuresOriginal[0:int(0.8 * len(images))]
            self.labels = labels[0:int(0.8 * len(images))]
            print(f"Loading celeba TRAIN dataset...\n{self.labels}")
        elif datanum == 2:
            self.imgs = images[int(0.8 * len(images)):]
            self.features = features[int(0.8 * len(images)):]
            self.featuresOriginal = featuresOriginal[int(0.8 * len(images)):]
            self.labels = labels[int(0.8 * len(images)):]
            print(f"Loading celeba TEST dataset...\n{self.labels}")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img, feature, label = self.imgs[index], self.features[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        if self.isOriginalFeature:
            return img, (feature, self.featuresOriginal[index]), label
        else:
            return img, feature, label


import torchvision.datasets as datasets


class LFW_old(Dataset):
    def __init__(self, transform=None, seed=666):
        olddataset = datasets.ImageFolder(root='/root/autodl-tmp/DATASET/lfw-deepfunneled', transform=transform)
        swapped_dict = {value: key for key, value in olddataset.class_to_idx.items()}
        dict_name = [np.where(olddataset.targets == i)[0] for i in np.unique(olddataset.targets)]
        dict_name = [dict_name[i] for i in range(len(dict_name)) if len(dict_name[i]) > 1]

        ori = []
        label = []
        for clz in dict_name:
            for idx in clz:
                ori.append(olddataset[idx][0])
                label.append(swapped_dict[olddataset[idx][1]])

        self.origins = ori
        self.labels = label

        # 打乱数据集
        np.random.seed(seed)
        perm = np.arange(len(self.origins))
        np.random.shuffle(perm)
        self.origins = np.array(self.origins)[perm]
        self.labels = np.array(self.labels)[perm]

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, index):
        img, label = self.origins[index], self.labels[index]
        return img, label


class LFW(Dataset):
    def __init__(self, dataPath, seed=666, transform=None, isOriginalFeature=False):
        assert 'lfw' in dataPath
        # 加载特征向量
        input = np.load(dataPath, allow_pickle=True)
        ori_feature255 = input['feature']
        ori = input['img']
        label = input['label']
        self.isOriginalFeature = isOriginalFeature
        self.dataPath = dataPath
        self.transform = transform

        v_min = ori_feature255.min(axis=0)
        v_max = ori_feature255.max(axis=0)
        ori_feature01 = (ori_feature255 - v_min) / (v_max - v_min)

        dict_name = [np.where(label == i)[0] for i in np.unique(label)]
        dict_name = [dict_name[i] for i in range(len(dict_name)) if
                     len(dict_name[i]) > 1]  # 删去只有一个样本的类别，以便tar和far计算时可以运算
        pass

        ORI = []
        LABEL = []
        FEATURE01 = []
        FEATURE255 = []
        for clz in dict_name:
            for idx in clz:
                ORI.append(ori[idx])
                LABEL.append(label[idx])
                FEATURE01.append(ori_feature01[idx])
                FEATURE255.append(ori_feature255[idx])

        self.origins = ORI
        self.labels = LABEL
        self.features_01 = FEATURE01
        self.features_255 = FEATURE255

        np.random.seed(seed)
        perm = np.arange(len(self.origins))
        np.random.shuffle(perm)
        self.origins = np.array(self.origins)[perm]
        self.labels = np.array(self.labels)[perm]
        self.features_01 = np.array(self.features_01)[perm]
        self.features_255 = np.array(self.features_255)[perm]

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, index):
        img, feature_01, label, feature_255 = self.origins[index], self.features_01[index], self.labels[index], \
                                              self.features_255[index]
        # if 'baidu' in self.dataPath:
        #     img = torch.tensor(img)
        # else:
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.isOriginalFeature:
            return img, (feature_01, feature_255), label
        else:
            return img, feature_01, label
class LFWORI(Dataset):
    def __init__(self, dataPath, seed=666, transform=None, isOriginalFeature=False):
        assert 'lfw' in dataPath
        # 加载特征向量
        input = np.load(dataPath, allow_pickle=True)
        ori_feature255 = input['feature']
        ori = input['img']
        label = input['label']
        self.isOriginalFeature = isOriginalFeature
        self.dataPath = dataPath
        self.transform = transform

        v_min = ori_feature255.min(axis=0)
        v_max = ori_feature255.max(axis=0)
        ori_feature01 = (ori_feature255 - v_min) / (v_max - v_min)

        dict_name = [np.where(label == i)[0] for i in np.unique(label)]
        dict_name = [dict_name[i] for i in range(len(dict_name)) if
                     len(dict_name[i]) > 1]  # 删去只有一个样本的类别，以便tar和far计算时可以运算
        pass

        ORI = []
        LABEL = []
        FEATURE01 = []
        FEATURE255 = []
        for clz in dict_name:
            for idx in clz:
                ORI.append(ori[idx])
                LABEL.append(label[idx])
                FEATURE01.append(ori_feature01[idx])
                FEATURE255.append(ori_feature255[idx])

        self.origins = ORI
        self.labels = LABEL
        self.features_01 = FEATURE01
        self.features_255 = FEATURE255

        np.random.seed(seed)
        perm = np.arange(len(self.origins))
        np.random.shuffle(perm)
        self.origins = np.array(self.origins)[perm]
        self.labels = np.array(self.labels)[perm]
        self.features_01 = np.array(self.features_01)[perm]
        self.features_255 = np.array(self.features_255)[perm]

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, index):
        img, feature_01, label, feature_255 = self.origins[index], self.features_01[index], self.labels[index], \
                                              self.features_255[index]
        # if 'baidu' in self.dataPath:
        #     img = torch.tensor(img)
        # else:
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.isOriginalFeature:
            return img, (feature_01, feature_255), label
        else:
            return img, feature_01, label

class FFHQ(Dataset):
    def __init__(self, dataPath, transform=None, seed=666, isOriginalFeature=False):
        self.transform = transform
        self.isOriginalFeature = isOriginalFeature

        # 加载特征向量
        assert 'ffhq' in dataPath
        input = np.load(dataPath, allow_pickle=True)
        ori = input['img']
        label = input['label']
        feature = input['feature']
        featureOriginal = copy.deepcopy(feature)

        v_min = feature.min(axis=0)
        v_max = feature.max(axis=0)
        feature = (feature - v_min) / (v_max - v_min)

        self.origins = ori
        self.labels = label
        self.features = feature
        self.featuresOriginal = featureOriginal

        np.random.seed(seed)
        perm = np.arange(len(self.origins))
        np.random.shuffle(perm)
        self.origins = np.array(self.origins)[perm]
        self.labels = np.array(self.labels)[perm]
        self.features = np.array(self.features)[perm]
        self.featuresOriginal = np.array(self.featuresOriginal)[perm]

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, index):
        img, feature_01, label, featureOri = self.origins[index], self.features[index], self.labels[index], \
                                             self.featuresOriginal[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.isOriginalFeature:
            return img, (feature_01, featureOri), label
        else:
            return img, feature_01, label


class Vggface(Dataset):
    def __init__(self, dataPath, transform=None, seed=666, isOriginalFeature=False):
        self.transform = transform
        self.isOriginalFeature = isOriginalFeature

        # 加载特征向量
        # assert 'vggface' in dataPath
        input = np.load(dataPath, allow_pickle=True)
        ori = input['img']
        label = input['label']
        feature = input['feature']
        featureOriginal = copy.deepcopy(feature)

        v_min = feature.min(axis=0)
        v_max = feature.max(axis=0)
        feature = (feature - v_min) / (v_max - v_min)

        self.origins = ori
        self.labels = label
        self.features = feature
        self.featuresOriginal = featureOriginal

        np.random.seed(seed)
        perm = np.arange(len(self.origins))
        np.random.shuffle(perm)
        self.origins = np.array(self.origins)[perm]
        self.labels = np.array(self.labels)[perm]
        self.features = np.array(self.features)[perm]
        self.featuresOriginal = np.array(self.featuresOriginal)[perm]

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, index):
        img, feature_01, label, featureOri = self.origins[index], self.features[index], self.labels[index], \
                                             self.featuresOriginal[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.isOriginalFeature:
            return img, (feature_01, featureOri), label
        else:
            return img, feature_01, label


class LFW500Done(Dataset):
    def __init__(self, dataDir, transform=None, seed=666, isOriginalFeature=False):
        self.transform = transform
        IMG, RECON, LABEL, TAG = [], [], [], []
        # 在./data/mydataset读取每个人名文件夹，需要读取ground_truth.png和
        outdir = os.listdir(dataDir)
        # 在dataset找出图片路径含有文本ground_truth的图片索引
        for tag in outdir:
            nowdir = os.path.join(dataDir, tag, 'result')
            imgpaths = os.listdir(nowdir)
            maxpath = ''
            maxval = -1
            for imgpath in imgpaths:
                if 'ground_truth' in imgpath:
                    IMG.append(Image.open(os.path.join(nowdir, imgpath)))
                    TAG.append(tag)
                    LABEL.append(tag[:-5])
                elif 'out' in imgpath:
                    if int(imgpath.split('_')[1]) > maxval:
                        maxval = int(imgpath.split('_')[1])
                        maxpath = os.path.join(nowdir, imgpath)
            RECON.append(Image.open(maxpath))

        classes = sorted(set(LABEL))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        for i in range(len(LABEL)):
            LABEL[i] = class_to_idx[LABEL[i]]

        self.recon = RECON
        self.ori = IMG
        self.label = LABEL

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        img, recon, label = self.ori[index], self.recon[index], self.label[index]

        if self.transform is not None:
            img = self.transform(img)
            recon = self.transform(recon)
            return img, recon, label


class LFW500(Dataset):
    def __init__(self, dataPath, seed=666, transform=None, isOriginalFeature=False):
        assert 'lfw' in dataPath
        # 加载特征向量
        input = np.load(dataPath, allow_pickle=True)
        ori_feature255 = input['feature']
        ori = input['img']
        label = input['label']
        tag = input['path']
        self.isOriginalFeature = isOriginalFeature
        self.dataPath = dataPath
        self.transform = transform

        v_min = ori_feature255.min(axis=0)
        v_max = ori_feature255.max(axis=0)
        ori_feature01 = (ori_feature255 - v_min) / (v_max - v_min)

        dict_name = [np.where(label == i)[0] for i in np.unique(label)]
        dict_name = [dict_name[i] for i in range(len(dict_name)) if
                     len(dict_name[i]) > 1]  # 删去只有一个样本的类别，以便tar和far计算时可以运算
        pass

        ORI = []
        LABEL = []
        FEATURE01 = []
        FEATURE255 = []
        TAG = []
        num_classes = 0
        for clz in dict_name:
            if len(clz) < 10:  # 取大于等于10张的类
                continue
            num_classes += 1
            if num_classes > 50:  # 取50个类
                break
            for idx in clz[:10]:  # 只取10张
                ORI.append(ori[idx])
                LABEL.append(label[idx])
                FEATURE01.append(ori_feature01[idx])
                FEATURE255.append(ori_feature255[idx])
                TAG.append(tag[idx].item().split('/')[-1].split('.')[0])

        self.origins = ORI
        self.labels = LABEL
        self.features_01 = FEATURE01
        self.features_255 = FEATURE255
        self.TAG = TAG

        # REVIEW
        # np.random.seed(seed)
        # perm = np.arange(len(self.origins))
        # np.random.shuffle(perm)
        # self.origins = np.array(self.origins)[perm]
        # self.labels = np.array(self.labels)[perm]
        # self.features_01 = np.array(self.features_01)[perm]
        # self.features_255 = np.array(self.features_255)[perm]

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, index):
        img, feature_01, label, feature_255 = self.origins[index], self.features_01[index], self.labels[index], \
                                              self.features_255[index]
        if 'baidu' in self.dataPath:
            img = torch.tensor(img)
        else:
            img = Image.fromarray(img)
            if self.transform is not None:
                img = self.transform(img)
        if self.isOriginalFeature:
            return img, (feature_01, feature_255), label
        else:
            return img, feature_01, label


class LFW14(Dataset):    # 前10张图

    def __init__(self, dataPath, seed=666, transform=None, isOriginalFeature=False):
        assert 'lfw' in dataPath
        # 加载特征向量
        input = np.load(dataPath, allow_pickle=True)
        ori_feature255 = input['feature']
        ori = input['img']
        label = input['label']
        tag = input['path']
        self.isOriginalFeature = isOriginalFeature
        self.dataPath = dataPath
        self.transform = transform

        v_min = ori_feature255.min(axis=0)
        v_max = ori_feature255.max(axis=0)
        ori_feature01 = (ori_feature255-v_min) / (v_max-v_min)

        dict_name = [np.where(label == i)[0] for i in np.unique(label)]
        # dict_name = [dict_name[i] for i in range(len(dict_name)) if len(dict_name[i]) > 1]               # 删去只有一个样本的类别，以便tar和far计算时可以运算
        dict_name = dict_name[2:]
        pass

        ORI = []
        LABEL = []
        FEATURE01 = []
        FEATURE255 = []
        TAG = []
        num_classes = 0
        for clz in dict_name:
            # if len(clz) < 10:               # 取大于等于10张的类
            #     continue
            num_classes += 1
            if num_classes > 10:               # 取前10个类
                break
            for idx in clz:
                ORI.append(ori[idx])
                LABEL.append(label[idx])
                FEATURE01.append(ori_feature01[idx])
                FEATURE255.append(ori_feature255[idx])
                TAG.append(tag[idx].item().split('/')[-1].split('.')[0])

        self.origins = ORI
        self.labels = LABEL
        self.features_01 = FEATURE01
        self.features_255 = FEATURE255
        self.TAG = TAG

        # REVIEW
        # np.random.seed(seed)
        # perm = np.arange(len(self.origins))
        # np.random.shuffle(perm)
        # self.origins = np.array(self.origins)[perm]
        # self.labels = np.array(self.labels)[perm]
        # self.features_01 = np.array(self.features_01)[perm]
        # self.features_255 = np.array(self.features_255)[perm]

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, index):
        img, feature_01, label, feature_255 = self.origins[index], self.features_01[index], self.labels[index], \
                                              self.features_255[index]
        if 'baidu' in self.dataPath:
            img = torch.tensor(img)
        else:
            img = Image.fromarray(img)
            if self.transform is not None:
                img = self.transform(img)
        if self.isOriginalFeature:
            return img, (feature_01, feature_255), label
        else:
            return img, feature_01, label


class FFHQ500(Dataset):
    def __init__(self, dataPath, transform=None, seed=666, isOriginalFeature=False):
        self.transform = transform
        self.isOriginalFeature = isOriginalFeature

        # 加载特征向量
        assert 'ffhq' in dataPath
        input = np.load(dataPath, allow_pickle=True)
        ori = input['img']
        label = input['label']
        feature = input['feature']
        path = input['path']
        featureOriginal = copy.deepcopy(feature)

        v_min = feature.min(axis=0)
        v_max = feature.max(axis=0)
        feature = (feature - v_min) / (v_max - v_min)

        dict_name = [np.where(label == i)[0] for i in np.unique(label)]
        dict_name = [dict_name[i] for i in range(len(dict_name)) if
                     len(dict_name[i]) > 1]  # 删去只有一个样本的类别，以便tar和far计算时可以运算
        # 打乱dict_name, 随机选择类
        np.random.seed(seed)
        perm = np.arange(len(dict_name))
        np.random.shuffle(perm)
        dict_name = np.array(dict_name)[perm]
        pass

        ORI = []
        LABEL = []
        FEATURE01 = []
        FEATURE255 = []
        PATH = []
        for clz in dict_name:
            if len(clz) < 2:  # 取大于等于2张的类
                continue
            for idx in clz[:2]:  # 只取2张
                ORI.append(ori[idx])
                LABEL.append(label[idx])
                FEATURE01.append(feature[idx])
                FEATURE255.append(featureOriginal[idx])
                if 'flip' in path[idx]:
                    PATH.append(path[idx].item().split('.')[0].split('/')[-1] + '_1')
                else:
                    PATH.append(path[idx].item().split('.')[0].split('/')[-1] + '_0')

            if len(ORI) >= 500:  # 取250*2张
                break

        self.origins = ORI
        self.labels = LABEL
        self.features = FEATURE01
        self.featuresOriginal = FEATURE255
        self.tag = PATH

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, index):
        img, feature_01, label, featureOri = self.origins[index], self.features[index], self.labels[index], \
                                             self.featuresOriginal[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.isOriginalFeature:
            return img, (feature_01, featureOri), label
        else:
            return img, feature_01, label


class FFHQ500Done(Dataset):
    def __init__(self, dataDir, transform=None, seed=666, isOriginalFeature=False):
        self.transform = transform
        IMG, RECON, LABEL, TAG = [], [], [], []
        # 在./data/mydataset读取每个人名文件夹，需要读取ground_truth.png和
        outdir = os.listdir(dataDir)
        # 在dataset找出图片路径含有文本ground_truth的图片索引
        for tag in outdir:
            nowdir = os.path.join(dataDir, tag, 'result')
            imgpaths = os.listdir(nowdir)
            maxpath = ''
            maxval = -1
            for imgpath in imgpaths:
                if 'ground_truth' in imgpath:
                    IMG.append(Image.open(os.path.join(nowdir, imgpath)))
                    TAG.append(tag)
                    LABEL.append(tag[:-5])
                elif 'out' in imgpath:
                    if int(imgpath.split('_')[1]) > maxval:
                        maxval = int(imgpath.split('_')[1])
                        maxpath = os.path.join(nowdir, imgpath)
            RECON.append(Image.open(maxpath))

        classes = sorted(set(LABEL))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        for i in range(len(LABEL)):
            LABEL[i] = class_to_idx[LABEL[i]]

        self.recon = RECON
        self.ori = IMG
        self.label = LABEL

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        img, recon, label = self.ori[index], self.recon[index], self.label[index]

        if self.transform is not None:
            img = self.transform(img)
            recon = self.transform(recon)
            return img, recon, label


class Vggface2(Dataset):
    def __init__(self, dataPath, imgdir, transform=None, seed=666, isOriginalFeature=False):
        self.transform = transform
        self.isOriginalFeature = isOriginalFeature
        self.imgdir = imgdir

        # 加载特征向量
        # assert 'vggface' in dataPath
        input = np.load(dataPath, allow_pickle=True)
        oripath = input['path']

        label = input['label']
        feature = input['feature']
        if isOriginalFeature:
            featureOriginal = feature[:]
        oripath = np.array(['/'.join([self.imgdir] + path.split('/')[1:]) for path in oripath])
        v_min = feature.min(axis=0)
        v_max = feature.max(axis=0)
        feature = (feature - v_min) / (v_max - v_min)

        self.originsPath = oripath
        self.labels = label
        self.features = feature
        if isOriginalFeature: self.featuresOriginal = featureOriginal

        np.random.seed(seed)
        perm = np.arange(len(self.originsPath))
        np.random.shuffle(perm)
        self.originsPath = np.array(self.originsPath)[perm]
        self.labels = np.array(self.labels)[perm]
        self.features = np.array(self.features)[perm]
        if isOriginalFeature: self.featuresOriginal = np.array(self.featuresOriginal)[perm]

    def __len__(self):
        return len(self.originsPath)

    def __getitem__(self, index):
        img, feature_01, label = Image.open(self.originsPath[index]), self.features[index], self.labels[
            index]


        if self.transform is not None:
            img = self.transform(img)

        if self.isOriginalFeature:
            featureOri = self.featuresOriginal[index]
            return img, (feature_01, featureOri), label
        else:
            return img, feature_01, label
