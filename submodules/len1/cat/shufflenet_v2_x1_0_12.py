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

    def forward(self, x277, x285):
        x286=torch.cat((x277, x285),dim=1)
        return x286

m = M().eval()
x277 = torch.randn(torch.Size([1, 232, 7, 7]))
x285 = torch.randn(torch.Size([1, 232, 7, 7]))
start = time.time()
output = m(x277, x285)
end = time.time()
print(end-start)
