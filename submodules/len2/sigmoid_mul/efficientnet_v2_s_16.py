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
        self.sigmoid16 = Sigmoid()

    def forward(self, x325, x321):
        x326=self.sigmoid16(x325)
        x327=operator.mul(x326, x321)
        return x327

m = M().eval()
x325 = torch.randn(torch.Size([1, 1536, 1, 1]))
x321 = torch.randn(torch.Size([1, 1536, 7, 7]))
start = time.time()
output = m(x325, x321)
end = time.time()
print(end-start)
