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
        self.conv2d67 = Conv2d(384, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d39 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d68 = Conv2d(128, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d40 = BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x195, x200):
        x201=operator.mul(x195, x200)
        x202=self.conv2d67(x201)
        x203=self.batchnorm2d39(x202)
        x204=self.conv2d68(x203)
        x205=self.batchnorm2d40(x204)
        return x205

m = M().eval()
x195 = torch.randn(torch.Size([1, 384, 14, 14]))
x200 = torch.randn(torch.Size([1, 384, 1, 1]))
start = time.time()
output = m(x195, x200)
end = time.time()
print(end-start)
