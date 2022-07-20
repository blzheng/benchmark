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
        self.conv2d117 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d117 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu112 = ReLU(inplace=True)
        self.conv2d118 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x386, x380):
        x387=self.conv2d117(x386)
        x388=self.batchnorm2d117(x387)
        x389=operator.add(x388, x380)
        x390=self.relu112(x389)
        x391=self.conv2d118(x390)
        return x391

m = M().eval()
x386 = torch.randn(torch.Size([1, 256, 14, 14]))
x380 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x386, x380)
end = time.time()
print(end-start)
