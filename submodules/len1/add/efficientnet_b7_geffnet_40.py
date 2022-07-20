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

    def forward(self, x693, x679):
        x694=operator.add(x693, x679)
        return x694

m = M().eval()
x693 = torch.randn(torch.Size([1, 384, 7, 7]))
x679 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x693, x679)
end = time.time()
print(end-start)