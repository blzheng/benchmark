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
        self.conv2d70 = Conv2d(1344, 2520, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d70 = BatchNorm2d(2520, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu69 = ReLU(inplace=True)

    def forward(self, x229, x239):
        x230=self.conv2d70(x229)
        x231=self.batchnorm2d70(x230)
        x240=operator.add(x231, x239)
        x241=self.relu69(x240)
        return x241

m = M().eval()
x229 = torch.randn(torch.Size([1, 1344, 14, 14]))
x239 = torch.randn(torch.Size([1, 2520, 7, 7]))
start = time.time()
output = m(x229, x239)
end = time.time()
print(end-start)
