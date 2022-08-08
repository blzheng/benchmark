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
        self.layernorm24 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)

    def forward(self, x531):
        x532=self.layernorm24(x531)
        return x532

m = M().eval()
x531 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x531)
end = time.time()
print(end-start)
