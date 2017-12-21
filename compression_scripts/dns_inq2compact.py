"""
convert dns_inq model to compact format (deep compression)
"""

import os, sys
import numpy as np

try: 
    caffe_root = os.environ["CAFFE_ROOT"]
except KeyError:
    print "Set system variable CAFFE_ROOT before running the script!"
    sys.exit(-1)

sys.path.append(caffe_root+"/python")
import caffe

help_ = """
Usage:
    dns_inq2compact.py <src_net.prototxt> <src_model.caffemodel> <target_net.prototxt> <target_model.caffemodel>
"""


if len(sys.argv) != 5:
    print help_
    sys.exit(-1)
else:
    f_src_net = sys.argv[0]
    f_src_model = sys.argv[1]
    # f_target_net = sys.argv[2]
    # f_target_model = sys.argv[3]
    f_target_model = sys.argv[3]

if not os.path.exists(f_src_net):
    print "Error: %s does not exist!"%(f_src_net)
    sys.exit()
elif not os.path.exists(f_src_model):
    print "Error: %s does not exist!"%(f_src_model)
    sys.exit()
elif not os.path.exists(f_target_net):
    print "Error: %s does not exist!"%(f_target_net)
    sys.exit()

caffe.set_mode_cpu()

net = caffe.Net(f_src_net, caffe.TEST, weight = src_model)
target_net = caffe.Net(f_target_net, caffe.TEST)

param_name_list = filter(lambda x: "conv" in x or "ip" in x or "fc" in x , net.params.keys())

n_bits = 4
num_quantum_exp = 7

def param_to_compact(wb):
    nz_idx = np.nonzero(wb)[0]      # nonzero() returns a tuple
    nz_wb = wb[nz_idx]              # get all the non-zero params
    nz_idx = np.concatenate((np.array(nz_idx[0]), np.diff(nz_idx))) 

    # tobe store
    param_idx = np.array([]).astype(np.int64)   
    # actural valued idx in param_idx
    param_idx_valued = np.array([]).astype(np.int64)    
    num_inserted = 0

    for i, idx in enumerate(nz_idx):
        if idx <= 2**bits:
            param_idx = np.append(param_idx, idx)
            param_idx_valued = np.append(param_idx_valued, i+num_inserted)
        else:
            num_to_insert = (idx-1)// (2**bits)
            num_inserted += num_to_insert
            param_idx = np.concatenate((param_idx, np.array([2**bits]*num_to_insert), np.array([idx - 2**bits*num_to_insert])))
            param_idx_valued = np.append(param_idx_valued, i+num_inserted)

    # check length
    if len(param_idx) != len(nz_wb)+num_inserted:
        print ("Error: length does not match (%d != %d)!"%(len(param_idx), len(nz_wb)+num_inserted))
        sys.exit()

    num_param = len(param_idx)
    wb_to_store = np.zeros(num_param)
    wb_to_store[param_idx_valued] = nz_wb
    max_exp = np.log2(np.max(np.fabs(nz_wb)))
    min_exp = max_exp - num_quantum_exp
    codebook = np.zeros(2**bits)

    #[0.0, -2^-7, -2^-6, ..., -2^-1, 2^-7, 2^-6, ..., 2^-1]
    for i in range(1, 2**(bits-1)): # 1 ~ 8
        codebook[i] = - np.exp2(min_exp + i -1)
        codebook[i+num_quantum_exp] = - np.exp2(min_exp + i -1)

    value_to_bits = {}
    for i, val in enumerate(codebook):
        value_to_bits[val] = i

    vfunc = lambda x: value_to_bits[x]
    # in case num_param is odd
    if num_param %2 == 1:
        np.append(wb_to_store, [0])

    wb_tmp = np.array(list(map(vfunc, wb_to_store))).astype(np.int8)
    wb_low_bit = wb_tmp[np.arange(0, len(wb_to_store), 2)]*2**bits + wb_tmp[np.arange(1, len(wb_to_store), 2)]
    param_idx_low_bit = param_idx[np.arange(0, len(param_idx), 2)]* 2**bits + param_idx[np.arange(1, len(param_idx), 2)]

    # (num_params, codebook, wb, idx)
    return num_param, codebook.astype(np.float32), wb_low_bit.astype(np.int8), param_idx_low_bit.astype(np.int8)

def save_data(f_out, num, codebook, lb_wb, lb_idx):
    np.array([num], dtype = int32).tofile(f_out)
    codebook.tofile(f_out)
    lb_wb.tofile(f_out)
    lb_idx.tofile(f_out)
    

f_out = open(f_target_model, 'w')

for param_name in param_name_list:

    w_inq = net.params[param_name][0].data.astype(np.float32).flatten()
    num_wb, codebook_wb, lb_wb, lb_idx = param_to_compact(w_inq)
    save_data(f_out, num_wb, codebook_wb, lb_wb, lb_idx)
    b_inq = net.params[param_name][1].data.astype(np.float32).flatten()
    num_wb, codebook_wb, lb_wb, lb_idx = param_to_compact(b_inq)
    save_data(f_out, num_wb, codebook_wb, lb_wb, lb_idx)


f_out.close()


