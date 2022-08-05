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
        self.layernorm20 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear18 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu8 = GELU(approximate='none')
        self.dropout16 = Dropout(p=0.0, inplace=False)

    def forward(self, x218):
        x219=self.layernorm20(x218)
        x220=self.linear18(x219)
        x221=self.gelu8(x220)
        x222=self.dropout16(x221)
        return x222

m = M().eval()
x218 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x218)
end = time.time()
print(end-start)
