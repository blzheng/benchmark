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
        self.layernorm14 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        self.linear43 = Linear(in_features=256, out_features=256, bias=True)

    def forward(self, x322):
        x323=self.layernorm14(x322)
        x324=self.linear43(x323)
        return x324

m = M().eval()
x322 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x322)
end = time.time()
print(end-start)
