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

    def forward(self, x319):
        x321=operator.getitem(x319, 1)
        return x321

m = M().eval()
x319 = (torch.randn((torch.Size([1, 352, 7, 7]), torch.randn(torch.Size([1, 352, 7, 7]), )
start = time.time()
output = m(x319)
end = time.time()
print(end-start)
