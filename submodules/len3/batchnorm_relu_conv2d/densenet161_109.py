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
        self.batchnorm2d110 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu110 = ReLU(inplace=True)
        self.conv2d110 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x389):
        x390=self.batchnorm2d110(x389)
        x391=self.relu110(x390)
        x392=self.conv2d110(x391)
        return x392

m = M().eval()
x389 = torch.randn(torch.Size([1, 192, 14, 14]))
start = time.time()
output = m(x389)
end = time.time()
print(end-start)
