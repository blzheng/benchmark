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

    def forward(self, x131, x132):
        x133=torch.matmul(x131, x132)
        return x133

m = M().eval()
x131 = torch.randn(torch.Size([1, 4, 384, 64]))
x132 = torch.randn(torch.Size([1, 4, 64, 384]))
start = time.time()
output = m(x131, x132)
end = time.time()
print(end-start)
