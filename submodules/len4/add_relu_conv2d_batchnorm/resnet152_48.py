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
        self.relu145 = ReLU(inplace=True)
        self.conv2d152 = Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d152 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x500, x492):
        x501=operator.add(x500, x492)
        x502=self.relu145(x501)
        x503=self.conv2d152(x502)
        x504=self.batchnorm2d152(x503)
        return x504

m = M().eval()
x500 = torch.randn(torch.Size([1, 2048, 7, 7]))
x492 = torch.randn(torch.Size([1, 2048, 7, 7]))
start = time.time()
output = m(x500, x492)
end = time.time()
print(end-start)
