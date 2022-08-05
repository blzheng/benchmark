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
        self.dropout12 = Dropout(p=0.1, inplace=False)
        self.layernorm8 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear24 = Linear(in_features=768, out_features=768, bias=True)

    def forward(self, x193, x190):
        x194=self.dropout12(x193)
        x195=operator.add(x194, x190)
        x196=self.layernorm8(x195)
        x197=self.linear24(x196)
        return x197

m = M().eval()
x193 = torch.randn(torch.Size([1, 384, 768]))
x190 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x193, x190)
end = time.time()
print(end-start)
