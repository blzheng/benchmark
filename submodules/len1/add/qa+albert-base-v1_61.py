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

    def forward(self, x397, x394):
        x398=operator.add(x397, x394)
        return x398

m = M().eval()
x397 = torch.randn(torch.Size([1, 384, 768]))
x394 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x397, x394)
end = time.time()
print(end-start)
