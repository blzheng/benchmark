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
        self.conv2d30 = Conv2d(12, 288, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid6 = Sigmoid()

    def forward(self, x92, x89):
        x93=self.conv2d30(x92)
        x94=self.sigmoid6(x93)
        x95=operator.mul(x94, x89)
        return x95

m = M().eval()
x92 = torch.randn(torch.Size([1, 12, 1, 1]))
x89 = torch.randn(torch.Size([1, 288, 56, 56]))
start = time.time()
output = m(x92, x89)
end = time.time()
print(end-start)
