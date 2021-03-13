import torch
import torch.nn as nn
import numpy as np
from IPython import embed

from .base_color import *

class ECCVGenerator(BaseColor):
    def __init__(self, norm_layer=nn.BatchNorm2d, hw_resize=256, model_name='eccv16'):
        super(ECCVGenerator, self).__init__()

        # Paper uses the general x2 rule for CNN
        self.hw_conv1 = hw_resize // 4  # 64
        self.hw_conv2 = hw_resize // 2  # 128
        self.hw_conv3 = hw_resize       # 256
        self.hw_conv4 = hw_resize * 2   # 512

        model1=[nn.Conv2d(1, self.hw_conv1, kernel_size=3, stride=1, padding=1, bias=True),]
        model1+=[nn.ReLU(True),]
        model1+=[nn.Conv2d(self.hw_conv1, self.hw_conv1, kernel_size=3, stride=2, padding=1, bias=True),]
        model1+=[nn.ReLU(True),]
        model1+=[norm_layer(self.hw_conv1),]

        model2=[nn.Conv2d(self.hw_conv1, self.hw_conv2, kernel_size=3, stride=1, padding=1, bias=True),]
        model2+=[nn.ReLU(True),]
        model2+=[nn.Conv2d(self.hw_conv2, self.hw_conv2, kernel_size=3, stride=2, padding=1, bias=True),]
        model2+=[nn.ReLU(True),]
        model2+=[norm_layer(self.hw_conv2),]

        model3=[nn.Conv2d(self.hw_conv2, self.hw_conv3, kernel_size=3, stride=1, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[nn.Conv2d(self.hw_conv3, self.hw_conv3, kernel_size=3, stride=1, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[nn.Conv2d(self.hw_conv3, self.hw_conv3, kernel_size=3, stride=2, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[norm_layer(self.hw_conv3),]

        model4=[nn.Conv2d(self.hw_conv3, self.hw_conv4, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[nn.Conv2d(self.hw_conv4, self.hw_conv4, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[nn.Conv2d(self.hw_conv4, self.hw_conv4, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[norm_layer(self.hw_conv4),]

        model5=[nn.Conv2d(self.hw_conv4, self.hw_conv4, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[nn.Conv2d(self.hw_conv4, self.hw_conv4, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[nn.Conv2d(self.hw_conv4, self.hw_conv4, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[norm_layer(self.hw_conv4),]

        model6=[nn.Conv2d(self.hw_conv4, self.hw_conv4, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[nn.Conv2d(self.hw_conv4, self.hw_conv4, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[nn.Conv2d(self.hw_conv4, self.hw_conv4, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[norm_layer(self.hw_conv4),]

        model7=[nn.Conv2d(self.hw_conv4, self.hw_conv4, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[nn.Conv2d(self.hw_conv4, self.hw_conv4, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[nn.Conv2d(self.hw_conv4, self.hw_conv4, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[norm_layer(self.hw_conv4),]

        model8=[nn.ConvTranspose2d(self.hw_conv4, self.hw_conv3, kernel_size=4, stride=2, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]
        model8+=[nn.Conv2d(self.hw_conv3, self.hw_conv3, kernel_size=3, stride=1, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]
        model8+=[nn.Conv2d(self.hw_conv3, self.hw_conv3, kernel_size=3, stride=1, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]

        model8+=[nn.Conv2d(self.hw_conv3, 313, kernel_size=1, stride=1, padding=0, bias=True),]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8 = nn.Sequential(*model8)

        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')

        # Model name
        self.name = model_name

    def forward(self, input_l):
        conv1_2 = self.model1(self.normalize_l(input_l))
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_3 = self.model8(conv7_3)
        out_reg = self.model_out(self.softmax(conv8_3))

        return self.unnormalize_ab(self.upsample4(out_reg))

def eccv16(pretrained=True, hw_resize=256, model_name='eccv16'):
    model = ECCVGenerator(hw_resize=hw_resize, model_name=model_name)
    if(pretrained):
        import torch.utils.model_zoo as model_zoo
        model.load_state_dict(
                model_zoo.load_url('https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth',
                model_dir='/Colorization/checkpoints/pretrained/',
                map_location='cpu',
                check_hash=True))
    return model
