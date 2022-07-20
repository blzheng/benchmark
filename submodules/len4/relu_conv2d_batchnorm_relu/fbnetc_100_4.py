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
        self.relu7 = ReLU(inplace=True)
        self.conv2d11 = Conv2d(24, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=24, bias=False)
        self.batchnorm2d11 = BatchNorm2d(24, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu8 = ReLU(inplace=True)

    def forward(self, x34):
        x35=self.relu7(x34)
        x36=self.conv2d11(x35)
        x37=self.batchnorm2d11(x36)
        x38=self.relu8(x37)
        return x38

m = M().eval()
x34 = torch.randn(torch.Size([1, 24, 56, 56]))
start = time.time()
output = m(x34)
end = time.time()
print(end-start)
