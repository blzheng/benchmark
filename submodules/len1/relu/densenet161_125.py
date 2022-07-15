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
        self.relu125 = ReLU(inplace=True)

    def forward(self, x444):
        x445=self.relu125(x444)
        return x445

m = M().eval()
x444 = torch.randn(torch.Size([1, 192, 7, 7]))
start = time.time()
output = m(x444)
end = time.time()
print(end-start)
