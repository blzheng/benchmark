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

    def forward(self, x397, x387):
        x398=operator.add(x397, x387)
        return x398

m = M().eval()
x397 = torch.randn(torch.Size([1, 768, 7, 7]))
x387 = torch.randn(torch.Size([1, 768, 7, 7]))
start = time.time()
output = m(x397, x387)
end = time.time()
print(end-start)
