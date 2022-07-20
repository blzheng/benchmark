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
        self.sigmoid10 = Sigmoid()
        self.conv2d53 = Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d31 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x159, x155):
        x160=self.sigmoid10(x159)
        x161=operator.mul(x160, x155)
        x162=self.conv2d53(x161)
        x163=self.batchnorm2d31(x162)
        return x163

m = M().eval()
x159 = torch.randn(torch.Size([1, 576, 1, 1]))
x155 = torch.randn(torch.Size([1, 576, 14, 14]))
start = time.time()
output = m(x159, x155)
end = time.time()
print(end-start)
