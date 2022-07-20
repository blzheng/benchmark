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

    def forward(self, x160, x156):
        x161=self.sigmoid10(x160)
        x162=operator.mul(x161, x156)
        return x162

m = M().eval()
x160 = torch.randn(torch.Size([1, 672, 1, 1]))
x156 = torch.randn(torch.Size([1, 672, 14, 14]))
start = time.time()
output = m(x160, x156)
end = time.time()
print(end-start)
