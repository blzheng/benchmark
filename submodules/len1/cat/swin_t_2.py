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

    def forward(self, x251, x252, x253, x254):
        x255=torch.cat([x251, x252, x253, x254], -1)
        return x255

m = M().eval()
x251 = torch.randn(torch.Size([1, 7, 7, 384]))
x252 = torch.randn(torch.Size([1, 7, 7, 384]))
x253 = torch.randn(torch.Size([1, 7, 7, 384]))
x254 = torch.randn(torch.Size([1, 7, 7, 384]))
start = time.time()
output = m(x251, x252, x253, x254)
end = time.time()
print(end-start)
