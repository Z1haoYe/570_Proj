import torch
import os 
import numpy as np
import torchvision.transforms.functional as tvF
from n2n_model import n2n
from Config import Config as conf
from skimage import io

from data_set_builder import Testinging_Dataset
from torch.utils.data import DataLoader

def test():
    device = torch.device('cuda:0')
    test_data = Testinging_Dataset(conf.data_path_test,conf.test_noise_param,conf.crop_img_size)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    model = n2n(input_channel=conf.img_channel,output_channel=conf.img_channel)
    model.load_state_dict(torch.load(conf.model_path_test))
    model.eval()
    model.cuda()
    result_dir = conf.denoised_dir
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    for batch_idx, (source,img_cropped) in enumerate(test_loader):
        test_img = tvF.to_pil_image(source.squeeze(0))
        ground_truth = img_cropped.squeeze(0).numpy().astype(np.uint8)
        source = source.to(device)
        denoised_img = model(source).detach().cpu()
        
        img_name = test_loader.dataset.image_list[batch_idx]
        
        denoised_result= tvF.to_pil_image(torch.clamp(denoised_img.squeeze(0), 0, 1))
        fname = os.path.splitext(img_name)[0]
        
        test_img.save(os.path.join(result_dir, f'{fname}-noisy.png'))
        denoised_result.save(os.path.join(result_dir, f'{fname}-denoised.png'))       
        io.imsave(os.path.join(result_dir, f'{fname}-ground_truth.png'),ground_truth)

def main():
    test()

if(__name__=="__main__"):
    main()