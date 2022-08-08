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
        self.relu19 = ReLU()
        self.conv2d26 = Conv2d(54, 216, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid4 = Sigmoid()
        self.conv2d27 = Conv2d(216, 216, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d17 = BatchNorm2d(216, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x79, x77):
        x80=self.relu19(x79)
        x81=self.conv2d26(x80)
        x82=self.sigmoid4(x81)
        x83=operator.mul(x82, x77)
        x84=self.conv2d27(x83)
        x85=self.batchnorm2d17(x84)
        return x85

m = M().eval()
x79 = torch.randn(torch.Size([1, 54, 1, 1]))
x77 = torch.randn(torch.Size([1, 216, 28, 28]))
start = time.time()
output = m(x79, x77)
end = time.time()
print(end-start)
