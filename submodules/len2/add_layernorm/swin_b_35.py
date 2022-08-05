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
        self.layernorm41 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)

    def forward(self, x448, x455):
        x456=operator.add(x448, x455)
        x457=self.layernorm41(x456)
        return x457

m = M().eval()
x448 = torch.randn(torch.Size([1, 14, 14, 512]))
x455 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x448, x455)
end = time.time()
print(end-start)
