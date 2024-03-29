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
        self.batchnorm2d113 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu109 = ReLU(inplace=True)
        self.conv2d114 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x374):
        x375=self.batchnorm2d113(x374)
        x376=self.relu109(x375)
        x377=self.conv2d114(x376)
        return x377

m = M().eval()
x374 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x374)
end = time.time()
print(end-start)
