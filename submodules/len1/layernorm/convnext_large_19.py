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
        self.layernorm19 = LayerNorm((768,), eps=1e-06, elementwise_affine=True)

    def forward(self, x230):
        x231=self.layernorm19(x230)
        return x231

m = M().eval()
x230 = torch.randn(torch.Size([1, 14, 14, 768]))
start = time.time()
output = m(x230)
end = time.time()
print(end-start)
