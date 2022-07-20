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
        self.conv2d81 = Conv2d(448, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d81 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x277):
        x278=self.conv2d81(x277)
        x279=self.batchnorm2d81(x278)
        x280=torch.nn.functional.relu(x279,inplace=True)
        return x280

m = M().eval()
x277 = torch.randn(torch.Size([1, 448, 5, 5]))
start = time.time()
output = m(x277)
end = time.time()
print(end-start)
