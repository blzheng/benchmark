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

    def forward(self, x253, x239):
        x254=operator.add(x253, (12, 64))
        x255=x239.view(x254)
        return x255

m = M().eval()
x253 = (1, 384, )
x239 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x253, x239)
end = time.time()
print(end-start)
