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
        self.dropout22 = Dropout(p=0.0, inplace=False)

    def forward(self, x298):
        x299=self.dropout22(x298)
        return x299

m = M().eval()
x298 = torch.randn(torch.Size([1, 7, 7, 3072]))
start = time.time()
output = m(x298)
end = time.time()
print(end-start)