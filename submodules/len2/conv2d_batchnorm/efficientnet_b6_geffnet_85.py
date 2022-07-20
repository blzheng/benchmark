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
        self.conv2d143 = Conv2d(200, 1200, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d85 = BatchNorm2d(1200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x427):
        x428=self.conv2d143(x427)
        x429=self.batchnorm2d85(x428)
        return x429

m = M().eval()
x427 = torch.randn(torch.Size([1, 200, 14, 14]))
start = time.time()
output = m(x427)
end = time.time()
print(end-start)
