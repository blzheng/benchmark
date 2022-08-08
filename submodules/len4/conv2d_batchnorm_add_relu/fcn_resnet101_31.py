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
        self.conv2d90 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d90 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu85 = ReLU(inplace=True)

    def forward(self, x298, x292):
        x299=self.conv2d90(x298)
        x300=self.batchnorm2d90(x299)
        x301=operator.add(x300, x292)
        x302=self.relu85(x301)
        return x302

m = M().eval()
x298 = torch.randn(torch.Size([1, 256, 28, 28]))
x292 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x298, x292)
end = time.time()
print(end-start)
