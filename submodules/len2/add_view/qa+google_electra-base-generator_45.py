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

    def forward(self, x501, x499):
        x502=operator.add(x501, (4, 64))
        x503=x499.view(x502)
        return x503

m = M().eval()
x501 = (1, 384, )
x499 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x501, x499)
end = time.time()
print(end-start)
