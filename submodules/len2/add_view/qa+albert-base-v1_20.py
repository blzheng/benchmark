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

    def forward(self, x266, x254):
        x267=operator.add(x266, (12, 64))
        x268=x254.view(x267)
        return x268

m = M().eval()
x266 = (1, 384, )
x254 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x266, x254)
end = time.time()
print(end-start)
