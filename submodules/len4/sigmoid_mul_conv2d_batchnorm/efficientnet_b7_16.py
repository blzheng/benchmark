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
        self.sigmoid16 = Sigmoid()
        self.conv2d81 = Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d47 = BatchNorm2d(80, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x251, x247):
        x252=self.sigmoid16(x251)
        x253=operator.mul(x252, x247)
        x254=self.conv2d81(x253)
        x255=self.batchnorm2d47(x254)
        return x255

m = M().eval()
x251 = torch.randn(torch.Size([1, 480, 1, 1]))
x247 = torch.randn(torch.Size([1, 480, 28, 28]))
start = time.time()
output = m(x251, x247)
end = time.time()
print(end-start)
