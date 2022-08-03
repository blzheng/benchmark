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
        self.linear23 = Linear(in_features=3072, out_features=768, bias=True)
        self.dropout12 = Dropout(p=0.1, inplace=False)
        self.layernorm8 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        self.linear24 = Linear(in_features=768, out_features=768, bias=True)

    def forward(self, x192, x190):
        x193=self.linear23(x192)
        x194=self.dropout12(x193)
        x195=operator.add(x194, x190)
        x196=self.layernorm8(x195)
        x197=self.linear24(x196)
        return x197

m = M().eval()
x192 = torch.randn(torch.Size([1, 384, 3072]))
x190 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x192, x190)
end = time.time()
print(end-start)
