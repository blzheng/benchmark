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
        self.conv2d29 = Conv2d(672, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d29 = BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu27 = ReLU(inplace=True)
        self.conv2d30 = Conv2d(672, 1344, kernel_size=(1, 1), stride=(2, 2), bias=False)

    def forward(self, x93, x87):
        x94=self.conv2d29(x93)
        x95=self.batchnorm2d29(x94)
        x96=operator.add(x87, x95)
        x97=self.relu27(x96)
        x98=self.conv2d30(x97)
        return x98

m = M().eval()
x93 = torch.randn(torch.Size([1, 672, 28, 28]))
x87 = torch.randn(torch.Size([1, 672, 28, 28]))
start = time.time()
output = m(x93, x87)
end = time.time()
print(end-start)
