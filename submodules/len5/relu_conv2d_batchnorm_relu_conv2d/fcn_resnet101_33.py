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
        self.relu49 = ReLU(inplace=True)
        self.conv2d55 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d55 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu52 = ReLU(inplace=True)
        self.conv2d56 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)

    def forward(self, x181):
        x182=self.relu49(x181)
        x183=self.conv2d55(x182)
        x184=self.batchnorm2d55(x183)
        x185=self.relu52(x184)
        x186=self.conv2d56(x185)
        return x186

m = M().eval()
x181 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x181)
end = time.time()
print(end-start)
