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
        self.conv2d72 = Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid14 = Sigmoid()

    def forward(self, x222, x219):
        x223=self.conv2d72(x222)
        x224=self.sigmoid14(x223)
        x225=operator.mul(x224, x219)
        return x225

m = M().eval()
x222 = torch.randn(torch.Size([1, 28, 1, 1]))
x219 = torch.randn(torch.Size([1, 672, 14, 14]))
start = time.time()
output = m(x222, x219)
end = time.time()
print(end-start)
