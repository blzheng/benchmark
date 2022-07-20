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

    def forward(self, x207):
        x208=torch.permute(x207, [0, 3, 1, 2])
        return x208

m = M().eval()
x207 = torch.randn(torch.Size([1, 7, 7, 768]))
start = time.time()
output = m(x207)
end = time.time()
print(end-start)
