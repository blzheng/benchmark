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
        self.conv2d26 = Conv2d(384, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d26 = BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x92):
        x93=self.conv2d26(x92)
        x94=self.batchnorm2d26(x93)
        return x94

m = M().eval()
x92 = torch.randn(torch.Size([1, 384, 28, 28]))
start = time.time()
output = m(x92)
end = time.time()
print(end-start)
