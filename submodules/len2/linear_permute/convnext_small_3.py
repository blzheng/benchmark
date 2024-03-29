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
        self.linear7 = Linear(in_features=768, out_features=192, bias=True)

    def forward(self, x51):
        x52=self.linear7(x51)
        x53=torch.permute(x52, [0, 3, 1, 2])
        return x53

m = M().eval()
x51 = torch.randn(torch.Size([1, 28, 28, 768]))
start = time.time()
output = m(x51)
end = time.time()
print(end-start)
