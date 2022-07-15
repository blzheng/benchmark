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

    def forward(self, x595):
        x596=torch.nn.functional.relu(x595,inplace=True)
        return x596

m = M().eval()
x595 = torch.randn(torch.Size([1, 1664, 7, 7]))
start = time.time()
output = m(x595)
end = time.time()
print(end-start)
