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

    def forward(self, x446, x438):
        x447=operator.add(x446, (12, 64))
        x448=x438.view(x447)
        return x448

m = M().eval()
x446 = (1, 384, )
x438 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x446, x438)
end = time.time()
print(end-start)
