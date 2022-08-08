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

    def forward(self, x252, x257):
        x258=x252.view(x257)
        return x258

m = M().eval()
x252 = torch.randn(torch.Size([1, 384, 768]))
x257 = (1, 384, 12, 64, )
start = time.time()
output = m(x252, x257)
end = time.time()
print(end-start)
