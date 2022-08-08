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
        self.relu97 = ReLU(inplace=True)
        self.conv2d103 = Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d103 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x339):
        x340=self.relu97(x339)
        x341=self.conv2d103(x340)
        x342=self.batchnorm2d103(x341)
        return x342

m = M().eval()
x339 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x339)
end = time.time()
print(end-start)
