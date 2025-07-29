import os
import numpy as np
import torch
# from model import SDNet as MyNet
import cv2
from dataset import TestData
import torch.utils.data as data
import torchvision.transforms as transforms

from model import FAMAFuse as MODEL

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS = 1e-8


def Mytest():
    EPOCHS = 50

    IMG_SAVE_DIR = "results/FAMANew/SPECT-MRI"  # CT PET SPECT
    MODEL_SAVE_PATH = "modelsave/FAMANew/SPECT"  # CT PET SPECT
    MODEL_SAVE_NAME ="mri_spect_weight.pth"

    os.makedirs(IMG_SAVE_DIR, exist_ok=True)
    model_path = MODEL_SAVE_PATH + '/' + str(EPOCHS) + '/'


    MODE = "test"
    TYPE = "SPECT-MRI" # SPECT PET CT
    IMG_TYPE1 = "SPECT/" # SPECT PET CT
    IMG_TYPE2 = "MRI/"
    TEST_DIR = f"D:/RESEARCH/IMAGE FUSION/FUSION DATASET/{TYPE}/{MODE}/"

    # net = fullModel()
    net = MODEL(1, 64)
    net.eval()
    net = net.to(DEVICE)
    net.load_state_dict(torch.load(model_path + MODEL_SAVE_NAME, map_location=torch.device('cpu')))

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((256, 256))])
    test_set = TestData(TEST_DIR,
                        IMG_TYPE1,
                        IMG_TYPE2,
                        transform)
    test_loader = data.DataLoader(test_set, batch_size=1, shuffle=False,
                                  num_workers=0, pin_memory=False)
    with torch.no_grad():
        if IMG_TYPE1 == 'CT/':
            for batch, [img_name, img1, img2] in enumerate(test_loader):  # CT-MRI Fusion
                print("test for image %s" % img_name[0])
                img1 = img1.to(DEVICE)
                img2 = img2.to(DEVICE)
                fused_img = net(img1, img2)
                fused_img = (fused_img - fused_img.min()) / (fused_img.max() - fused_img.min()) * 255.
                fused_img = fused_img.cpu().numpy().squeeze()
                # fused_img = fused_img.astype(np.uint8)
                cv2.imwrite('%s/%s' % (IMG_SAVE_DIR, img_name[0]), fused_img)
        else:
            for batch, [img_name, img1_Y, img2, img1_CrCb] in enumerate(test_loader):  # PET/SPECT-MRI Fusion

                print("test for image %s" % img_name[0])

                img1_Y = img1_Y.to(DEVICE)
                img2 = img2.to(DEVICE)
                fused_img_Y= net(img1_Y, img2)

                fused_img_Y = (fused_img_Y - fused_img_Y.min()) / (fused_img_Y.max() - fused_img_Y.min()) * 255.
                fused_img_Y = fused_img_Y.cpu().numpy()

                fused_img = np.concatenate((fused_img_Y, img1_CrCb), axis=1).squeeze()
                fused_img = np.transpose(fused_img, (1, 2, 0))
                fused_img = fused_img.astype(np.uint8)
                fused_img = cv2.cvtColor(fused_img, cv2.COLOR_YCrCb2BGR)

                cv2.imwrite('%s/%s' % (IMG_SAVE_DIR, img_name[0]), fused_img)
    print('test results in ./%s/' % IMG_SAVE_DIR)
    print('Finish!')

if __name__ == '__main__':
    Mytest()