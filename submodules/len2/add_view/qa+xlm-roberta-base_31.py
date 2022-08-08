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

    def forward(self, x352, x350):
        x353=operator.add(x352, (768,))
        x354=x350.view(x353)
        return x354

m = M().eval()
x352 = (1, 384, )
x350 = torch.randn(torch.Size([1, 384, 12, 64]))
start = time.time()
output = m(x352, x350)
end = time.time()
print(end-start)
