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
        self.layernorm20 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear60 = Linear(in_features=768, out_features=768, bias=True)

    def forward(self, x446, x442):
        x447=operator.add(x446, x442)
        x448=self.layernorm20(x447)
        x449=self.linear60(x448)
        return x449

m = M().eval()
x446 = torch.randn(torch.Size([1, 384, 768]))
x442 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x446, x442)
end = time.time()
print(end-start)
