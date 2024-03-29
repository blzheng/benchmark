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
        self.linear34 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu16 = GELU(approximate='none')

    def forward(self, x403):
        x404=self.linear34(x403)
        x405=self.gelu16(x404)
        return x405

m = M().eval()
x403 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x403)
end = time.time()
print(end-start)
