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
        self.dropout21 = Dropout(p=0.1, inplace=False)

    def forward(self, x320):
        x321=self.dropout21(x320)
        return x321

m = M().eval()
x320 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x320)
end = time.time()
print(end-start)
