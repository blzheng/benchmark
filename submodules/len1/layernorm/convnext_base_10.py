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
        self.layernorm10 = LayerNorm((512,), eps=1e-06, elementwise_affine=True)

    def forward(self, x131):
        x132=self.layernorm10(x131)
        return x132

m = M().eval()
x131 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x131)
end = time.time()
print(end-start)