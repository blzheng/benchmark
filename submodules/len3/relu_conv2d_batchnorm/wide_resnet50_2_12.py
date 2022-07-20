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
        self.relu13 = ReLU(inplace=True)
        self.conv2d16 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d16 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x50):
        x51=self.relu13(x50)
        x52=self.conv2d16(x51)
        x53=self.batchnorm2d16(x52)
        return x53

m = M().eval()
x50 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x50)
end = time.time()
print(end-start)
