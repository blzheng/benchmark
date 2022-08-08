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

    def forward(self, x184, x182):
        x185=operator.add(x184, (768,))
        x186=x182.view(x185)
        return x186

m = M().eval()
x184 = (1, 384, )
x182 = torch.randn(torch.Size([1, 384, 12, 64]))
start = time.time()
output = m(x184, x182)
end = time.time()
print(end-start)
