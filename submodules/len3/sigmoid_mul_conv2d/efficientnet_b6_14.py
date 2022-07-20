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
        self.sigmoid14 = Sigmoid()
        self.conv2d72 = Conv2d(432, 72, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x222, x218):
        x223=self.sigmoid14(x222)
        x224=operator.mul(x223, x218)
        x225=self.conv2d72(x224)
        return x225

m = M().eval()
x222 = torch.randn(torch.Size([1, 432, 1, 1]))
x218 = torch.randn(torch.Size([1, 432, 28, 28]))
start = time.time()
output = m(x222, x218)
end = time.time()
print(end-start)
