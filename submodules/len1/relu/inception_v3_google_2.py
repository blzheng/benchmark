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

    def forward(self, x21):
        x22=torch.nn.functional.relu(x21,inplace=True)
        return x22

m = M().eval()
x21 = torch.randn(torch.Size([1, 64, 109, 109]))
start = time.time()
output = m(x21)
end = time.time()
print(end-start)
