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
        self.conv2d57 = Conv2d(224, 896, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid10 = Sigmoid()

    def forward(self, x178, x175):
        x179=self.conv2d57(x178)
        x180=self.sigmoid10(x179)
        x181=operator.mul(x180, x175)
        return x181

m = M().eval()
x178 = torch.randn(torch.Size([1, 224, 1, 1]))
x175 = torch.randn(torch.Size([1, 896, 14, 14]))
start = time.time()
output = m(x178, x175)
end = time.time()
print(end-start)
