import torch
import torch.nn as nn
# from torchvision.models.utils import load_state_dict_from_url
from torchvision._internally_replaced_utils import load_state_dict_from_url

__all__ = ['MobileNetV1', 'mobilenet_v1']

model_urls = {
    'mobilenet_v1': 'https://storage.googleapis.com/models-hao/mobilenet-v1-ssd-mp-0_675.pth',
}

class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(  3,  32, 2), 
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.model(x)
        x = x.reshape(-1, 1024)
        x = self.fc(x)
        return x

def mobilenet_v1(pretrained=False, progress=True, **kwargs):
    model = MobileNetV1(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v1'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
