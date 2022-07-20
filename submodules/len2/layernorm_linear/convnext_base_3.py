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
        self.layernorm3 = LayerNorm((256,), eps=1e-06, elementwise_affine=True)
        self.linear6 = Linear(in_features=256, out_features=1024, bias=True)

    def forward(self, x48):
        x49=self.layernorm3(x48)
        x50=self.linear6(x49)
        return x50

m = M().eval()
x48 = torch.randn(torch.Size([1, 28, 28, 256]))
start = time.time()
output = m(x48)
end = time.time()
print(end-start)
