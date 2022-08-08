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

    def forward(self, x438, x447):
        x448=x438.view(x447)
        return x448

m = M().eval()
x438 = torch.randn(torch.Size([1, 384, 768]))
x447 = (1, 384, 12, 64, )
start = time.time()
output = m(x438, x447)
end = time.time()
print(end-start)
