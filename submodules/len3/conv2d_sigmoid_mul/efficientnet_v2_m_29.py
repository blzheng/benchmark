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
        self.conv2d172 = Conv2d(76, 1824, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid29 = Sigmoid()

    def forward(self, x551, x548):
        x552=self.conv2d172(x551)
        x553=self.sigmoid29(x552)
        x554=operator.mul(x553, x548)
        return x554

m = M().eval()
x551 = torch.randn(torch.Size([1, 76, 1, 1]))
x548 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x551, x548)
end = time.time()
print(end-start)
