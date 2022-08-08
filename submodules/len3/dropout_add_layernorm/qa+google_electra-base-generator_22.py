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
        self.dropout35 = Dropout(p=0.1, inplace=False)
        self.layernorm23 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)

    def forward(self, x524, x491):
        x525=self.dropout35(x524)
        x526=operator.add(x525, x491)
        x527=self.layernorm23(x526)
        return x527

m = M().eval()
x524 = torch.randn(torch.Size([1, 384, 256]))
x491 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x524, x491)
end = time.time()
print(end-start)
