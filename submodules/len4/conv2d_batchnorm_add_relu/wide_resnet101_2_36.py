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
        self.conv2d103 = Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d103 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu97 = ReLU(inplace=True)

    def forward(self, x338, x332):
        x339=self.conv2d103(x338)
        x340=self.batchnorm2d103(x339)
        x341=operator.add(x340, x332)
        x342=self.relu97(x341)
        return x342

m = M().eval()
x338 = torch.randn(torch.Size([1, 1024, 7, 7]))
x332 = torch.randn(torch.Size([1, 2048, 7, 7]))
start = time.time()
output = m(x338, x332)
end = time.time()
print(end-start)
