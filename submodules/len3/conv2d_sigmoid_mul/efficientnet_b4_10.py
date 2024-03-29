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
        self.conv2d52 = Conv2d(14, 336, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid10 = Sigmoid()

    def forward(self, x160, x157):
        x161=self.conv2d52(x160)
        x162=self.sigmoid10(x161)
        x163=operator.mul(x162, x157)
        return x163

m = M().eval()
x160 = torch.randn(torch.Size([1, 14, 1, 1]))
x157 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x160, x157)
end = time.time()
print(end-start)
