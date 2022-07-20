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
        self.relu33 = ReLU(inplace=True)
        self.conv2d45 = Conv2d(1392, 1392, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=6, bias=False)
        self.batchnorm2d29 = BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu34 = ReLU(inplace=True)

    def forward(self, x139):
        x140=self.relu33(x139)
        x141=self.conv2d45(x140)
        x142=self.batchnorm2d29(x141)
        x143=self.relu34(x142)
        return x143

m = M().eval()
x139 = torch.randn(torch.Size([1, 1392, 14, 14]))
start = time.time()
output = m(x139)
end = time.time()
print(end-start)
