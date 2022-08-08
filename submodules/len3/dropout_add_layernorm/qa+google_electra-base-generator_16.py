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
        self.dropout26 = Dropout(p=0.1, inplace=False)
        self.layernorm17 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)

    def forward(self, x398, x365):
        x399=self.dropout26(x398)
        x400=operator.add(x399, x365)
        x401=self.layernorm17(x400)
        return x401

m = M().eval()
x398 = torch.randn(torch.Size([1, 384, 256]))
x365 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x398, x365)
end = time.time()
print(end-start)
