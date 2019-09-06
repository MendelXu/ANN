import torch
from block.apnb import APNB
from block.base_oc_block import BaseOC_Module
from block.null_block import NullBlock
import time
import sys

def write2csv(table,name,column_size = None):
    print(table)
    with open(name+'.csv','w') as f:
        for t in table:
            if column_size is not None and len(t) < column_size:
                t = t + [' ']*(column_size - len(t))
            f.write((',').join(t))
            f.write('\n')
def filter_blank(lst):
    out = []
    for l in lst:
        if l != '':
            out.append(l)
    return out
def filter_end(lst,w):
    if isinstance(lst,list):
        out = []
        for l in lst:
            out.append(l.strip(w))
    else:
        out = float(lst.strip(w))
    return out

def test(model,inputs = [1,2048,128,256],times = 20):
    model = model.cuda()
    model.eval()
    with torch.no_grad():
        with torch.autograd.profiler.profile(use_cuda=True) as prof:
            for i in range(times):
                x = torch.randn(*inputs).cuda()
                out = model(x)
                del x 
                del out
                torch.cuda.empty_cache()
    res = prof.key_averages().table()
    res = res.strip().split('\n')
    table = []
    keys = res[1].strip().split('     ')
    keys = [r.strip() for r in keys]    
    keys = filter_blank(keys[0:3:2])
    table.append(keys)
    for i in range(3,len(res)):
        if res[i].strip() == '':
            continue
        tmp = res[i].strip().split()
        tmp = [t.strip() for t in tmp]
        tmp = filter_blank(tmp)
        tmp = tmp[0:1] + [str(filter_end(tmp[-1],'us')/times)]
        table.append(tmp)
    return table     
    
def test_multi(size): 
    model = NullBlock()
    test(model,inputs = size)
    del model
    torch.cuda.empty_cache()
    model = APNB(2048,2048,256,256,dropout=0.05)
    table= [['APNB']]
    table.extend(test(model,inputs=size))
    write2csv(table,'pspoc')
    del model
    torch.cuda.empty_cache()
    
    model= BaseOC_Module(2048,2048,256,256,dropout=0.05)
    table.append(['NonLocal'])
    table.extend(test(model,inputs=size))
    del model
    torch.cuda.empty_cache()
    write2csv(table,'summary',column_size=max(len(l) for l in table))


sizes=[[1,2048,96,96],[1,2048,128,128],[1,2048,128,256]]
 
test_multi(sizes[int(sys.argv[1])-1])
