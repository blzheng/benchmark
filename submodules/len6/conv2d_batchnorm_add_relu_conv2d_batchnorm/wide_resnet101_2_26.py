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
        self.conv2d75 = Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d75 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu70 = ReLU(inplace=True)
        self.conv2d76 = Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d76 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x246, x240):
        x247=self.conv2d75(x246)
        x248=self.batchnorm2d75(x247)
        x249=operator.add(x248, x240)
        x250=self.relu70(x249)
        x251=self.conv2d76(x250)
        x252=self.batchnorm2d76(x251)
        return x252

m = M().eval()
x246 = torch.randn(torch.Size([1, 512, 14, 14]))
x240 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x246, x240)
end = time.time()
print(end-start)
