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
        self.sigmoid5 = Sigmoid()

    def forward(self, x97, x93):
        x98=self.sigmoid5(x97)
        x99=operator.mul(x98, x93)
        return x99

m = M().eval()
x97 = torch.randn(torch.Size([1, 216, 1, 1]))
x93 = torch.randn(torch.Size([1, 216, 28, 28]))
start = time.time()
output = m(x97, x93)
end = time.time()
print(end-start)
