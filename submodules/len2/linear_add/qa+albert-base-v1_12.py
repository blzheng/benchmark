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
        self.linear6 = Linear(in_features=3072, out_features=768, bias=True)

    def forward(self, x470, x468):
        x471=self.linear6(x470)
        x472=operator.add(x471, x468)
        return x472

m = M().eval()
x470 = torch.randn(torch.Size([1, 384, 3072]))
x468 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x470, x468)
end = time.time()
print(end-start)
