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
        self.conv2d56 = Conv2d(320, 80, kernel_size=(1, 1), stride=(1, 1))
        self.relu43 = ReLU()
        self.conv2d57 = Conv2d(80, 320, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid10 = Sigmoid()
        self.conv2d58 = Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d36 = BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x176, x175):
        x177=self.conv2d56(x176)
        x178=self.relu43(x177)
        x179=self.conv2d57(x178)
        x180=self.sigmoid10(x179)
        x181=operator.mul(x180, x175)
        x182=self.conv2d58(x181)
        x183=self.batchnorm2d36(x182)
        return x183

m = M().eval()
x176 = torch.randn(torch.Size([1, 320, 1, 1]))
x175 = torch.randn(torch.Size([1, 320, 14, 14]))
start = time.time()
output = m(x176, x175)
end = time.time()
print(end-start)
