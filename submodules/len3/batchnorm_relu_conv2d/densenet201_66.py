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
        self.batchnorm2d67 = BatchNorm2d(704, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu67 = ReLU(inplace=True)
        self.conv2d67 = Conv2d(704, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x239):
        x240=self.batchnorm2d67(x239)
        x241=self.relu67(x240)
        x242=self.conv2d67(x241)
        return x242

m = M().eval()
x239 = torch.randn(torch.Size([1, 704, 14, 14]))
start = time.time()
output = m(x239)
end = time.time()
print(end-start)
