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
        self.conv2d97 = Conv2d(1184, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d98 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu98 = ReLU(inplace=True)
        self.conv2d98 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x346):
        x347=self.conv2d97(x346)
        x348=self.batchnorm2d98(x347)
        x349=self.relu98(x348)
        x350=self.conv2d98(x349)
        return x350

m = M().eval()
x346 = torch.randn(torch.Size([1, 1184, 14, 14]))
start = time.time()
output = m(x346)
end = time.time()
print(end-start)
