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
        self.conv2d9 = Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d9 = BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x44):
        x45=self.conv2d9(x44)
        x46=self.batchnorm2d9(x45)
        x47=torch.nn.functional.relu(x46,inplace=True)
        return x47

m = M().eval()
x44 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x44)
end = time.time()
print(end-start)
