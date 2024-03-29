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

    def forward(self, x447, x443):
        x448=operator.add(x447, x443)
        return x448

m = M().eval()
x447 = torch.randn(torch.Size([1, 384, 256]))
x443 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x447, x443)
end = time.time()
print(end-start)
