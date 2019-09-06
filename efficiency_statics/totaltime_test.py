import torch
from block.apnb import APNB
from block.base_oc_block import BaseOC_Module
from block.null_block import NullBlock
import time
import sys

def forward(model,inputs):
    out = model(inputs)
    torch.cuda.synchronize()

def test(model,inputs=[1,2048,128,256],times=20):
    #print(model)
    model = model.cuda()
    model.eval()
    time_seq = []
    gpumem = 0
    with torch.no_grad():
        for i in range(times):
            x = torch.randn(*inputs).cuda()
            start = time.time()
            forward(model,x)
            end = time.time()
            torch.cuda.empty_cache()
            
            time_seq.append(end-start)
            gpumem = max(gpumem,torch.cuda.max_memory_allocated()/1024/1024)
        avg_time = sum(time_seq)/times
    return gpumem , avg_time

def test_multi(size): 
    model = NullBlock()
    base_gpumem , base_time = test(model,inputs=size)
    del model
    torch.cuda.empty_cache()
    model = APNB(2048,2048,256,256,dropout=0.05)
    
    psp_gpumem , psp_time = test(model,inputs=size)
    psp_gpumem = psp_gpumem - base_gpumem
    psp_time = psp_time - base_time
    del model
    torch.cuda.empty_cache()
    
    model= BaseOC_Module(2048,2048,256,256,dropout=0.05)
    
    nl_gpumem,nl_time = test(model,inputs=size)
    nl_gpumem = nl_gpumem - base_gpumem
    nl_time = nl_time -base_time
    del model
    torch.cuda.empty_cache()
    
    print('Test on Inputs of ',size)
    print(' \tGPU Mem(MB)\tTime(ms)')
    print('psp\t%.2f     \t%.3f'%(psp_gpumem,psp_time*1000))
    print('nl\t%.2f      \t%.3f'%(nl_gpumem,nl_time*1000))


sizes=[[1,2048,96,96],[1,2048,128,128],[1,2048,128,256]]
 
test_multi(sizes[int(sys.argv[1])-1])
