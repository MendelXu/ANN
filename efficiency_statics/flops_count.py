import torch
from block.apnb import APNB
from block.base_oc_block import BaseOC_Module
from block.null_block import NullBlock
import time
import sys
#from thop import profile
from torchstat import stat as profile

def test_multi(size):
    model = NullBlock()
    profile(model,size[1:])
    del model
    torch.cuda.empty_cache()
    model = APNB(2048, 2048, 256, 256, dropout=0.05)
    profile(model,size[1:])
    del model
    torch.cuda.empty_cache()
    model = BaseOC_Module(2048, 2048, 256, 256, dropout=0.05)
    profile(model,size[1:])
    del model
    torch.cuda.empty_cache()


sizes = [[1, 2048, 96, 96], [1, 2048, 128, 128], [1, 2048, 128, 256]]

test_multi(sizes[int(sys.argv[1])-1])
