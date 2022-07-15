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

    def forward(self, x313):
        x314=torch.nn.functional.relu(x313,inplace=True)
        return x314

m = M().eval()
x313 = torch.randn(torch.Size([1, 384, 5, 5]))
start = time.time()
output = m(x313)
end = time.time()
print(end-start)
