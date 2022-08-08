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

    def forward(self, x493, x496):
        x497=x493.view(x496)
        return x497

m = M().eval()
x493 = torch.randn(torch.Size([1, 384, 256]))
x496 = (1, 384, 4, 64, )
start = time.time()
output = m(x493, x496)
end = time.time()
print(end-start)
