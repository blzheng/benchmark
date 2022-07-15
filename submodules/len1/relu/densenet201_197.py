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
        self.relu197 = ReLU(inplace=True)

    def forward(self, x696):
        x697=self.relu197(x696)
        return x697

m = M().eval()
x696 = torch.randn(torch.Size([1, 128, 7, 7]))
start = time.time()
output = m(x696)
end = time.time()
print(end-start)
