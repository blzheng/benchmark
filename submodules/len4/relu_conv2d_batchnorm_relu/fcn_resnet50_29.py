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
        self.relu43 = ReLU(inplace=True)
        self.conv2d50 = Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d50 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu46 = ReLU(inplace=True)

    def forward(self, x163):
        x164=self.relu43(x163)
        x165=self.conv2d50(x164)
        x166=self.batchnorm2d50(x165)
        x167=self.relu46(x166)
        return x167

m = M().eval()
x163 = torch.randn(torch.Size([1, 2048, 28, 28]))
start = time.time()
output = m(x163)
end = time.time()
print(end-start)
