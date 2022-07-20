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
        self.batchnorm2d57 = BatchNorm2d(816, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu57 = ReLU(inplace=True)
        self.conv2d57 = Conv2d(816, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d58 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x204):
        x205=self.batchnorm2d57(x204)
        x206=self.relu57(x205)
        x207=self.conv2d57(x206)
        x208=self.batchnorm2d58(x207)
        return x208

m = M().eval()
x204 = torch.randn(torch.Size([1, 816, 14, 14]))
start = time.time()
output = m(x204)
end = time.time()
print(end-start)
