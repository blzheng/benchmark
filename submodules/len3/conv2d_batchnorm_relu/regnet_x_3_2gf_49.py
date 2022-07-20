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
        self.conv2d77 = Conv2d(1008, 1008, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d77 = BatchNorm2d(1008, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu73 = ReLU(inplace=True)

    def forward(self, x251):
        x252=self.conv2d77(x251)
        x253=self.batchnorm2d77(x252)
        x254=self.relu73(x253)
        return x254

m = M().eval()
x251 = torch.randn(torch.Size([1, 1008, 7, 7]))
start = time.time()
output = m(x251)
end = time.time()
print(end-start)
