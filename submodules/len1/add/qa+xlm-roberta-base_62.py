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

    def forward(self, x394):
        x395=operator.add(x394, (768,))
        return x395

m = M().eval()
x394 = (1, 384, )
start = time.time()
output = m(x394)
end = time.time()
print(end-start)
