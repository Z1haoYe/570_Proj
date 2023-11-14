import torch
import torch.nn as nn

class n2n (nn.Module):
    def __init__(self, input_channel = 3, output_channel = 3):
        super(n2n, self).__init__()
        self.pool1 = nn.Sequential(
            nn.Conv2d(input_channel,48,3,stride=1,padding=1),
            nn.LeakyReLU(negative_slope = 0.05),
            nn.Conv2d(48,48,3,stride=1,padding=1), 
            nn.LeakyReLU(negative_slope = 0.05),
            nn.MaxPool2d(2))
        self.pool2 = nn.Sequential(
            nn.Conv2d(48,48,3,stride=1,padding=1),
            nn.LeakyReLU(negative_slope = 0.05),
            nn.MaxPool2d(2))
        self.pool3 = nn.Sequential(
            nn.Conv2d(48,48,3,stride=1,padding=1),
            nn.LeakyReLU(negative_slope = 0.05),
            nn.MaxPool2d(2))
        self.pool4 = nn.Sequential(
            nn.Conv2d(48,48,3,stride=1,padding=1),
            nn.LeakyReLU(negative_slope = 0.05),
            nn.MaxPool2d(2))
        self.pool5 = nn.Sequential(
            nn.Conv2d(48,48,3,stride=1,padding=1),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1))
        self.up1 = nn.Sequential(
            nn.Conv2d(96,96,3,stride=1,padding=1),
            nn.LeakyReLU(negative_slope = 0.05),
            nn.Conv2d(96,96,3,stride=1,padding=1),
            nn.LeakyReLU(negative_slope = 0.05),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
        ## deconv4a_4b_upsample3
        self.up2 = nn.Sequential(
            nn.Conv2d(144,96,3,stride=1,padding=1),
            nn.LeakyReLU(negative_slope = 0.05),
            nn.Conv2d(96,96,3,stride=1,padding=1),
            nn.LeakyReLU(negative_slope = 0.05),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1)) 
        ##  deconv3a_3b_upsample2
        self.up3 = nn.Sequential(
            nn.Conv2d(144,96,3,stride=1,padding=1),
            nn.LeakyReLU(negative_slope = 0.05),
            nn.Conv2d(96,96,3,stride=1,padding=1),
            nn.LeakyReLU(negative_slope = 0.05),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
        self.up4 = nn.Sequential (
            nn.Conv2d(96 + input_channel,64,3,stride=1,padding=1),
            nn.LeakyReLU(negative_slope = 0.05),
            nn.Conv2d(64,32,3,stride=1,padding=1),
            nn.LeakyReLU(negative_slope = 0.05)) 
        self.output_layer = nn.Conv2d(32,output_channel,3,stride=1,padding=1)
        self._init_weights()

    def forward(self,x):
            '''
            forward function
            '''
            pool_out1 = self.pool1(x)
            pool_out2 = self.pool2(pool_out1)
            pool_out3 = self.pool3(pool_out2)
            pool_out4 = self.pool4(pool_out3)


            up_out1 = self.pool5(pool_out4)
            concat1 = torch.cat((up_out1,pool_out3),dim=1)
            up_out2 = self.up1(concat1)
            concat2 = torch.cat((up_out2,pool_out2),dim=1)
            up_out3 = self.up2(concat2)
            concat3 = torch.cat((up_out3,pool_out1),dim=1)
            up_out4 = self.up3(concat3)
            concat4 = torch.cat((up_out4,x),dim =1)
            up_out5 = self.up4(concat4)
            output = self.output_layer(up_out5)
            return output
        
    def _init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)