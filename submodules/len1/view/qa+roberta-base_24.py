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

    def forward(self, x282, x285):
        x286=x282.view(x285)
        return x286

m = M().eval()
x282 = torch.randn(torch.Size([1, 384, 768]))
x285 = (1, 384, 12, 64, )
start = time.time()
output = m(x282, x285)
end = time.time()
print(end-start)
