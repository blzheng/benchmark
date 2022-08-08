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
        self.dropout2 = Dropout(p=0.1, inplace=False)
        self.layernorm1 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)

    def forward(self, x391, x362):
        x392=self.dropout2(x391)
        x393=operator.add(x362, x392)
        x394=self.layernorm1(x393)
        return x394

m = M().eval()
x391 = torch.randn(torch.Size([1, 384, 768]))
x362 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x391, x362)
end = time.time()
print(end-start)