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
        self.conv2d117 = Conv2d(726, 2904, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid22 = Sigmoid()
        self.conv2d118 = Conv2d(2904, 2904, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d72 = BatchNorm2d(2904, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x370, x367, x361):
        x371=self.conv2d117(x370)
        x372=self.sigmoid22(x371)
        x373=operator.mul(x372, x367)
        x374=self.conv2d118(x373)
        x375=self.batchnorm2d72(x374)
        x376=operator.add(x361, x375)
        return x376

m = M().eval()
x370 = torch.randn(torch.Size([1, 726, 1, 1]))
x367 = torch.randn(torch.Size([1, 2904, 14, 14]))
x361 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x370, x367, x361)
end = time.time()
print(end-start)
