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
        self.relu104 = ReLU(inplace=True)
        self.conv2d135 = Conv2d(888, 888, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d83 = BatchNorm2d(888, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x411, x425):
        x426=operator.add(x411, x425)
        x427=self.relu104(x426)
        x428=self.conv2d135(x427)
        x429=self.batchnorm2d83(x428)
        return x429

m = M().eval()
x411 = torch.randn(torch.Size([1, 888, 7, 7]))
x425 = torch.randn(torch.Size([1, 888, 7, 7]))
start = time.time()
output = m(x411, x425)
end = time.time()
print(end-start)
