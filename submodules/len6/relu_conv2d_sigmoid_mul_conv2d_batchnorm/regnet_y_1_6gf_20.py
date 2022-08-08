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
        self.relu83 = ReLU()
        self.conv2d107 = Conv2d(84, 336, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid20 = Sigmoid()
        self.conv2d108 = Conv2d(336, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d66 = BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x337, x335):
        x338=self.relu83(x337)
        x339=self.conv2d107(x338)
        x340=self.sigmoid20(x339)
        x341=operator.mul(x340, x335)
        x342=self.conv2d108(x341)
        x343=self.batchnorm2d66(x342)
        return x343

m = M().eval()
x337 = torch.randn(torch.Size([1, 84, 1, 1]))
x335 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x337, x335)
end = time.time()
print(end-start)
