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

    def forward(self, x239, x254):
        x255=x239.view(x254)
        return x255

m = M().eval()
x239 = torch.randn(torch.Size([1, 384, 768]))
x254 = (1, 384, 12, 64, )
start = time.time()
output = m(x239, x254)
end = time.time()
print(end-start)
