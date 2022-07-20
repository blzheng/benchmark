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
        self.gelu34 = GELU(approximate='none')
        self.linear69 = Linear(in_features=3072, out_features=768, bias=True)

    def forward(self, x403):
        x404=self.gelu34(x403)
        x405=self.linear69(x404)
        return x405

m = M().eval()
x403 = torch.randn(torch.Size([1, 7, 7, 3072]))
start = time.time()
output = m(x403)
end = time.time()
print(end-start)
