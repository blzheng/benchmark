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

    def forward(self, x183, x186):
        x187=x183.view(x186)
        return x187

m = M().eval()
x183 = torch.randn(torch.Size([1, 384, 4, 64]))
x186 = (1, 384, 256, )
start = time.time()
output = m(x183, x186)
end = time.time()
print(end-start)
