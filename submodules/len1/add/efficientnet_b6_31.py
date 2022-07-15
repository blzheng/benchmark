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

    def forward(self, x589, x574):
        x590=operator.add(x589, x574)
        return x590

m = M().eval()
x589 = torch.randn(torch.Size([1, 344, 7, 7]))
x574 = torch.randn(torch.Size([1, 344, 7, 7]))
start = time.time()
output = m(x589, x574)
end = time.time()
print(end-start)
