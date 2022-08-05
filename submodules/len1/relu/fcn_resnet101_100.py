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
        self.relu100 = ReLU()

    def forward(self, x346):
        x347=self.relu100(x346)
        return x347

m = M().eval()
x346 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x346)
end = time.time()
print(end-start)
