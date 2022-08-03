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

    def forward(self, x341):
        x343=operator.getitem(x341, 1)
        return x343

m = M().eval()
x341 = (torch.randn((torch.Size([1, 232, 7, 7]), torch.randn(torch.Size([1, 232, 7, 7]), )
start = time.time()
output = m(x341)
end = time.time()
print(end-start)
