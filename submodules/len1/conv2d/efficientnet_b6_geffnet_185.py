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
        self.conv2d185 = Conv2d(2064, 86, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x553):
        x554=self.conv2d185(x553)
        return x554

m = M().eval()
x553 = torch.randn(torch.Size([1, 2064, 1, 1]))
start = time.time()
output = m(x553)
end = time.time()
print(end-start)
