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
        self.conv2d129 = Conv2d(1696, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d130 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu130 = ReLU(inplace=True)
        self.conv2d130 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x458):
        x459=self.conv2d129(x458)
        x460=self.batchnorm2d130(x459)
        x461=self.relu130(x460)
        x462=self.conv2d130(x461)
        return x462

m = M().eval()
x458 = torch.randn(torch.Size([1, 1696, 14, 14]))
start = time.time()
output = m(x458)
end = time.time()
print(end-start)
