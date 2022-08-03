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
        self.dropout29 = Dropout(p=0.1, inplace=False)
        self.layernorm19 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)

    def forward(self, x439, x406):
        x440=self.dropout29(x439)
        x441=operator.add(x440, x406)
        x442=self.layernorm19(x441)
        return x442

m = M().eval()
x439 = torch.randn(torch.Size([1, 384, 768]))
x406 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x439, x406)
end = time.time()
print(end-start)
