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

    def forward(self, x51):
        x52=torch.nn.functional.relu(x51,inplace=True)
        return x52

m = M().eval()
x51 = torch.randn(torch.Size([1, 32, 25, 25]))
start = time.time()
output = m(x51)
end = time.time()
print(end-start)