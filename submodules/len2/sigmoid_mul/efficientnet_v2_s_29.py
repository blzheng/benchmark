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
        self.sigmoid29 = Sigmoid()

    def forward(self, x533, x529):
        x534=self.sigmoid29(x533)
        x535=operator.mul(x534, x529)
        return x535

m = M().eval()
x533 = torch.randn(torch.Size([1, 1536, 1, 1]))
x529 = torch.randn(torch.Size([1, 1536, 7, 7]))
start = time.time()
output = m(x533, x529)
end = time.time()
print(end-start)
