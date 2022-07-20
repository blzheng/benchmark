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
        self.sigmoid13 = Sigmoid()

    def forward(self, x229, x225):
        x230=self.sigmoid13(x229)
        x231=operator.mul(x230, x225)
        return x231

m = M().eval()
x229 = torch.randn(torch.Size([1, 440, 1, 1]))
x225 = torch.randn(torch.Size([1, 440, 7, 7]))
start = time.time()
output = m(x229, x225)
end = time.time()
print(end-start)
