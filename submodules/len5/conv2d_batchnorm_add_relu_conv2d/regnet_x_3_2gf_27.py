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
        self.conv2d76 = Conv2d(1008, 1008, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d76 = BatchNorm2d(1008, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu72 = ReLU(inplace=True)
        self.conv2d77 = Conv2d(1008, 1008, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x247, x241):
        x248=self.conv2d76(x247)
        x249=self.batchnorm2d76(x248)
        x250=operator.add(x241, x249)
        x251=self.relu72(x250)
        x252=self.conv2d77(x251)
        return x252

m = M().eval()
x247 = torch.randn(torch.Size([1, 1008, 7, 7]))
x241 = torch.randn(torch.Size([1, 1008, 7, 7]))
start = time.time()
output = m(x247, x241)
end = time.time()
print(end-start)
