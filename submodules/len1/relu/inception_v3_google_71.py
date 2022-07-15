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

    def forward(self, x246):
        x247=torch.nn.functional.relu(x246,inplace=True)
        return x247

m = M().eval()
x246 = torch.randn(torch.Size([1, 320, 5, 5]))
start = time.time()
output = m(x246)
end = time.time()
print(end-start)
