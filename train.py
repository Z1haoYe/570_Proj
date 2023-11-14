import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
import os
import torchvision.transforms.functional as tvF
# from unet_model import UNet\
from n2n_model import n2n
from Config import Config as conf
import time
from data_set_builder import Training_Dataset
from torch.utils.data import Dataset, DataLoader


def train():
    device = torch.device('cuda: 0')
    train_data = Training_Dataset(conf.data_path_train,conf.gaussian_noise_param,conf.crop_img_size)
    data_len = len(train_data)
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True,num_workers=4)
    model = n2n(input_channel =conf.img_channel,output_channel=conf.img_channel)
    model = model.cuda()
    optimizer = Adam(model.parameters(), lr = conf.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=True)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    model.train()
    start = time.time()
    for epoch in range(conf.max_epoch):
        print('Epoch {}/{}'.format(epoch, conf.max_epoch - 1))
        print('*' * 10)
        total_loss = 0.0
        for batch_idx, (source, target) in enumerate(train_loader):
            source = source.cuda()
            tartget = target.cude()
            optimizer.zero_grad()
            _source = model(source)
            loss = nn.MSELoss(_source, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()*source.size(0)
            print(f'loss: {loss.item():.3f}, batch idx: {batch_idx}')
        scheduler.step()
        avg_loss = total_loss /data_len
        print('{} Loss: {:.4f}'.format('current '+ str(epoch), avg_loss))
    save_nn(model, epoch + 1)
    end = time.time()
    time_elapsed = end - start
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))      

def save_nn(model, epoch):
    file_name = '/n2n_{}.pth'.format(epoch)
    path = conf.data_path_checkpoint
    if not  os.path.exists(path):
        os.mkdir(path)
    print('Saving epoch {}\n',format(epoch))
    torch.save(model.state_dict(), path + file_name)

def main():
    train()


if(__name__=="__main__"):
    main()