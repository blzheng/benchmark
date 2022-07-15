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
        self.relu156 = ReLU(inplace=True)

    def forward(self, x553):
        x554=self.relu156(x553)
        return x554

m = M().eval()
x553 = torch.randn(torch.Size([1, 1472, 7, 7]))
start = time.time()
output = m(x553)
end = time.time()
print(end-start)
