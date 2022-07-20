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
        self.conv2d53 = Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid10 = Sigmoid()

    def forward(self, x159, x156):
        x160=self.conv2d53(x159)
        x161=self.sigmoid10(x160)
        x162=operator.mul(x161, x156)
        return x162

m = M().eval()
x159 = torch.randn(torch.Size([1, 28, 1, 1]))
x156 = torch.randn(torch.Size([1, 672, 14, 14]))
start = time.time()
output = m(x159, x156)
end = time.time()
print(end-start)
