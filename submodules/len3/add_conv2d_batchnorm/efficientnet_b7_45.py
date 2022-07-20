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
        self.conv2d262 = Conv2d(640, 3840, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d156 = BatchNorm2d(3840, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x824, x809):
        x825=operator.add(x824, x809)
        x826=self.conv2d262(x825)
        x827=self.batchnorm2d156(x826)
        return x827

m = M().eval()
x824 = torch.randn(torch.Size([1, 640, 7, 7]))
x809 = torch.randn(torch.Size([1, 640, 7, 7]))
start = time.time()
output = m(x824, x809)
end = time.time()
print(end-start)
