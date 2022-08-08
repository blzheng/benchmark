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

    def forward(self, x372, x375):
        x376=x372.view(x375)
        return x376

m = M().eval()
x372 = torch.randn(torch.Size([1, 384, 768]))
x375 = (1, 384, 12, 64, )
start = time.time()
output = m(x372, x375)
end = time.time()
print(end-start)
