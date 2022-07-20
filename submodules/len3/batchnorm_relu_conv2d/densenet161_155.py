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
        self.batchnorm2d156 = BatchNorm2d(2112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu156 = ReLU(inplace=True)
        self.conv2d156 = Conv2d(2112, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x552):
        x553=self.batchnorm2d156(x552)
        x554=self.relu156(x553)
        x555=self.conv2d156(x554)
        return x555

m = M().eval()
x552 = torch.randn(torch.Size([1, 2112, 7, 7]))
start = time.time()
output = m(x552)
end = time.time()
print(end-start)
