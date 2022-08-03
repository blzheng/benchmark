import torch
from torch import tensor
import torch.nn as nn
from torch.nn import *
import torchvision
import torchvision.models as models
from torchvision.ops.stochastic_depth import stochastic_depth
import time
import builtins
import operator

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.conv2d35 = Conv2d(128, 128, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
        self.batchnorm2d35 = BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d36 = Conv2d(128, 128, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
        self.batchnorm2d36 = BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d37 = Conv2d(128, 128, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
        self.batchnorm2d37 = BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x127):
        x128=torch.nn.functional.relu(x127,inplace=True)
        x129=self.conv2d35(x128)
        x130=self.batchnorm2d35(x129)
        x131=torch.nn.functional.relu(x130,inplace=True)
        x132=self.conv2d36(x131)
        x133=self.batchnorm2d36(x132)
        x134=torch.nn.functional.relu(x133,inplace=True)
        x135=self.conv2d37(x134)
        x136=self.batchnorm2d37(x135)
        return x136

m = M().eval()
x127 = torch.randn(torch.Size([1, 128, 12, 12]))
start = time.time()
output = m(x127)
end = time.time()
print(end-start)
