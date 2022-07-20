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
        self.batchnorm2d26 = BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu26 = ReLU(inplace=True)
        self.conv2d26 = Conv2d(480, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d27 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x94):
        x95=self.batchnorm2d26(x94)
        x96=self.relu26(x95)
        x97=self.conv2d26(x96)
        x98=self.batchnorm2d27(x97)
        return x98

m = M().eval()
x94 = torch.randn(torch.Size([1, 480, 28, 28]))
start = time.time()
output = m(x94)
end = time.time()
print(end-start)
