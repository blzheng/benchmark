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
        self.conv2d136 = Conv2d(896, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d137 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu137 = ReLU(inplace=True)

    def forward(self, x484):
        x485=self.conv2d136(x484)
        x486=self.batchnorm2d137(x485)
        x487=self.relu137(x486)
        return x487

m = M().eval()
x484 = torch.randn(torch.Size([1, 896, 7, 7]))
start = time.time()
output = m(x484)
end = time.time()
print(end-start)
