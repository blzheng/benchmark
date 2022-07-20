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
        self.conv2d48 = Conv2d(1344, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d48 = BatchNorm2d(1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu45 = ReLU(inplace=True)

    def forward(self, x155, x149):
        x156=self.conv2d48(x155)
        x157=self.batchnorm2d48(x156)
        x158=operator.add(x149, x157)
        x159=self.relu45(x158)
        return x159

m = M().eval()
x155 = torch.randn(torch.Size([1, 1344, 14, 14]))
x149 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x155, x149)
end = time.time()
print(end-start)
