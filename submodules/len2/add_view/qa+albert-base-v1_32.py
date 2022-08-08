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

    def forward(self, x414, x402):
        x415=operator.add(x414, (12, 64))
        x416=x402.view(x415)
        return x416

m = M().eval()
x414 = (1, 384, )
x402 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x414, x402)
end = time.time()
print(end-start)
