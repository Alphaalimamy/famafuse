import os
import cv2
import random
import torch.utils.data as data
import torchvision.transforms as transforms

class TrainData(data.Dataset):
    def __init__(self, dir_train, img_type1,img_type2, transform=None):
        super(TrainData, self).__init__()
        self.patch_size = 256
        self.dir_prefix = dir_train
        self.img_type1 = img_type1
        self.img_type2 = img_type2
        self.transform = transform

        self.img1_dir = os.listdir(self.dir_prefix + self.img_type1)
        self.img2_dir = os.listdir(self.dir_prefix + self.img_type2)


    def __len__(self):
        assert len(self.img1_dir) == len(self.img2_dir)
        return len(self.img1_dir)

    def __getitem__(self, index):
        if self.img_type1 == 'CT/':
            img1 = cv2.imread(self.dir_prefix + self.img_type1 + self.img1_dir[index], cv2.IMREAD_GRAYSCALE)
        else:
            img1 = cv2.imread(self.dir_prefix + self.img_type1 + self.img1_dir[index])
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)
            img1 = img1[:, :, 0:1]
            img1 = img1.squeeze()
        img2 = cv2.imread(self.dir_prefix + self.img_type2 + self.img2_dir[index], cv2.IMREAD_GRAYSCALE)

        img1_p, img2_p = self.get_patch(img1, img2)
        if self.transform:
            img1_p = self.transform(img1_p)
            img2_p = self.transform(img2_p)

        return img1_p, img2_p  # 1,256,256

    def get_patch(self, img1, img2):
        h, w = img1.shape[:2]
        stride = self.patch_size

        x = random.randint(0, w - stride)
        y = random.randint(0, h - stride)

        img1_p = img1[y:y + stride, x:x + stride]
        img2_p = img2[y:y + stride, x:x + stride]

        return img1_p, img2_p


class TestData(data.Dataset):
    def __init__(self,dir_test,img_type1,img_type2, transform=None):
        super(TestData, self).__init__()
        self.transform = transform
        self.dir_prefix = dir_test
        self.img_type1 = img_type1
        self.img_type2 = img_type2

        self.img1_dir = os.listdir(self.dir_prefix + img_type1)
        self.img2_dir = os.listdir(self.dir_prefix + img_type2)

    def __getitem__(self, index):
        img_name = str(self.img1_dir[index])
        if self.img_type1 == 'CT/':
            img1 = cv2.imread(self.dir_prefix + self.img_type1 + self.img1_dir[index], cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(self.dir_prefix + self.img_type2 + self.img2_dir[index], cv2.IMREAD_GRAYSCALE)
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            return img_name, img1, img2  # img1[YCrCb]:3,256,256  img2[Gray]:1,256,256
        else:
            img1 = cv2.imread(self.dir_prefix + self.img_type1 + self.img1_dir[index])
            # img1 = cv2.imread(self.dir_prefix + args.img_type1 + self.img1_dir[index], cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(self.dir_prefix + self.img_type2 + self.img2_dir[index], cv2.IMREAD_GRAYSCALE)

            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)  # CT/PET/SPECT 256,256,3

            img1_Y = img1[:, :, 0:1]
            img1_CrCb = img1[:, :, 1:3].transpose(2, 0, 1)

            if self.transform:
                img1_Y = self.transform(img1_Y)
                img2 = self.transform(img2)

            return img_name, img1_Y, img2, img1_CrCb  # img1[YCrCb]:3,256,256  img2[Gray]:1,256,256

    def __len__(self):
        assert len(self.img1_dir) == len(self.img2_dir)
        return len(self.img1_dir)


if __name__ == "__main__":
    MODE = "test"
    TYPE = "PET-MRI" # SPECT-MRI PET-MRI
    TRAIN_DIR = f"D:/RESEARCH/IMAGE FUSION/FUSION DATASET/{TYPE}/{MODE}/"
    TEMP = "./temp/PET-MRI"
    IMG_TYPE1 = "PET/"
    IMG_TYPE2 = "MRI/"

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((256, 256))])

    train_set = TestData(
        TRAIN_DIR,
        IMG_TYPE1,
        IMG_TYPE2,
        transform=transform)
    train_loader = data.DataLoader(train_set,
                                   batch_size=1,
                                   shuffle=True,
                                   drop_last=True,
                                   num_workers=2,
                                   pin_memory=True)


    print(f"train_loader: {len(train_loader)}")
