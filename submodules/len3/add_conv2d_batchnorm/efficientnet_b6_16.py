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
        self.conv2d103 = Conv2d(144, 864, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d61 = BatchNorm2d(864, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x321, x306):
        x322=operator.add(x321, x306)
        x323=self.conv2d103(x322)
        x324=self.batchnorm2d61(x323)
        return x324

m = M().eval()
x321 = torch.randn(torch.Size([1, 144, 14, 14]))
x306 = torch.randn(torch.Size([1, 144, 14, 14]))
start = time.time()
output = m(x321, x306)
end = time.time()
print(end-start)
