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

    def forward(self, x538, x523):
        x539=operator.add(x538, x523)
        return x539

m = M().eval()
x538 = torch.randn(torch.Size([1, 256, 7, 7]))
x523 = torch.randn(torch.Size([1, 256, 7, 7]))
start = time.time()
output = m(x538, x523)
end = time.time()
print(end-start)
