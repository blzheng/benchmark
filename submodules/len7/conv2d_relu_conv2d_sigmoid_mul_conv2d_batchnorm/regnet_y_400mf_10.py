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
        self.conv2d57 = Conv2d(440, 52, kernel_size=(1, 1), stride=(1, 1))
        self.relu43 = ReLU()
        self.conv2d58 = Conv2d(52, 440, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid10 = Sigmoid()
        self.conv2d59 = Conv2d(440, 440, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d37 = BatchNorm2d(440, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x178, x177):
        x179=self.conv2d57(x178)
        x180=self.relu43(x179)
        x181=self.conv2d58(x180)
        x182=self.sigmoid10(x181)
        x183=operator.mul(x182, x177)
        x184=self.conv2d59(x183)
        x185=self.batchnorm2d37(x184)
        return x185

m = M().eval()
x178 = torch.randn(torch.Size([1, 440, 1, 1]))
x177 = torch.randn(torch.Size([1, 440, 7, 7]))
start = time.time()
output = m(x178, x177)
end = time.time()
print(end-start)
