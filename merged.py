import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import glob
import cv2

from data_loader import RescaleT
from data_loader import CenterCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset
from model import BASNet

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn

def save_output(image_name, pred, d_dir):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np * 255).convert('RGB')
    img_name = image_name.split("/")[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir + imidx + '.png')

def merge_images(original_img_path, masked_img_path, result_path):
    original_img = cv2.imread(original_img_path)
    masked_img = cv2.imread(masked_img_path, cv2.IMREAD_GRAYSCALE)
    
    _, binary_mask = cv2.threshold(masked_img, 128, 255, cv2.THRESH_BINARY)

    # Create a copy of the original image with an alpha channel
    result = cv2.cvtColor(original_img, cv2.COLOR_BGR2BGRA)

    # Set the non-masked part of the result to have a transparent background
    result[binary_mask == 0] = [0, 0, 0, 0]  # 4th channel (alpha) is set to 0 for transparency

    # Save the result with transparency in PNG format
    cv2.imwrite(result_path, result)

def process_images(image_dir, prediction_dir, result_dir, model_dir):
    img_name_list = glob.glob(image_dir + '*.*')

    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list, lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(256), ToTensorLab(flag=0)]))
    test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=1)

    print("...load BASNet...")
    net = BASNet(3, 1)
    net.load_state_dict(torch.load(model_dir))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    for i_test, data_test in enumerate(test_salobj_dataloader):
        print("inferencing:", img_name_list[i_test].split("/")[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, _, _, _, _, _, _, _ = net(inputs_test)

        pred = d1[:, 0, :, :]
        pred = normPRED(pred)

        save_output(img_name_list[i_test], pred, prediction_dir)

        original_img_path = img_name_list[i_test]
        masked_img_path = os.path.join(prediction_dir, f'clicked{i_test}.png')
        result_path = os.path.join(result_dir, f'result{i_test}.png')
        merge_images(original_img_path, masked_img_path, result_path)

        result = cv2.imread(result_path)

        # cv2.imshow('Result', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        del d1

if __name__ == '__main__':
    process_images(
        '/home/druglord/Documents/Jan-25th/Virtual_Photo_Studio/bnet/click/',
        '/home/druglord/Documents/Jan-25th/Virtual_Photo_Studio/bnet/mask/',
        '/home/druglord/Documents/Jan-25th/Virtual_Photo_Studio/bnet/result/',
        './saved_models/basnet_bsi/basnet.pth'
    )