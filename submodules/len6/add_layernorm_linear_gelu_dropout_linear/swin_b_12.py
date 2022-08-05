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
        self.layernorm28 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear26 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu12 = GELU(approximate='none')
        self.dropout24 = Dropout(p=0.0, inplace=False)
        self.linear27 = Linear(in_features=2048, out_features=512, bias=True)

    def forward(self, x295, x309):
        x310=operator.add(x295, x309)
        x311=self.layernorm28(x310)
        x312=self.linear26(x311)
        x313=self.gelu12(x312)
        x314=self.dropout24(x313)
        x315=self.linear27(x314)
        return x315

m = M().eval()
x295 = torch.randn(torch.Size([1, 14, 14, 512]))
x309 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x295, x309)
end = time.time()
print(end-start)
