"""
convert dns_inq model to compact format (deep compression)
"""

import os, sys
import numpy as np
import pprint as pp
import datetime
import time

try: 
    caffe_root = os.environ["CAFFE_ROOT"]
except KeyError:
    print "Set system variable CAFFE_ROOT before running the script!"
    sys.exit(-1)

sys.path.append(caffe_root+"/python")
import caffe

help_ = """
Usage:
    dns_inq2compact.py <src_net.prototxt> <src_model.caffemodel> <target_model.caffemodel>
"""


if len(sys.argv) != 4:
    print help_
    sys.exit(-1)
else:
    f_src_net = sys.argv[1]
    f_src_model = sys.argv[2]
    f_target_model = sys.argv[3]
    # f_target_model = sys.argv[3]

if not os.path.exists(f_src_net):
    print "Error: %s does not exist!"%(f_src_net)
    sys.exit()
elif not os.path.exists(f_src_model):
    print "Error: %s does not exist!"%(f_src_model)
    sys.exit()
"""
elif not os.path.exists(f_target_model):
    print "Error: %s does not exist!"%(f_target_model)
    sys.exit()
"""
caffe.set_mode_cpu()

# print "f_src_net: %s"%f_src_net
# print "f_src_model: %s"%f_src_model
net = caffe.Net(f_src_net, caffe.TEST, weights=f_src_model)
# target_net = caffe.Net(f_target_net, caffe.TEST)

param_name_list = filter(lambda x: "conv" in x or "ip" in x or "fc" in x , net.params.keys())
print "\n\nlayer list: ", param_name_list
np.set_printoptions(threshold='nan')

bits = 4
num_quantum_exp = 7
check_layer = -1

def param_to_compact(wb, flag=-1):
    nz_idx = np.nonzero(wb)[0]      # nonzero() returns a tuple
    
    if flag ==check_layer:
        print "origin nz_idx:\n", nz_idx
    
    nz_wb = wb[nz_idx]              # get all the non-zero params
    if len(nz_wb) == 0:
        return 0, np.zeros(2**bits).astype(np.float32), np.array([]).astype(np.uint8), np.array([]).astype(np.uint8)

    nz_idx = nz_idx +1              # assume the idx starts from 1
    # diff the nz_idx
    nz_idx = np.concatenate((np.array([nz_idx[0]]), np.diff(nz_idx))) 
     
    if flag == check_layer:
        print "diff nz_idx:\n", nz_idx
     
    le_16_idx = np.nonzero( nz_idx <= 2**bits)[0]
    le_16_value = nz_idx[le_16_idx]
    
    gt_16_idx = np.nonzero( nz_idx > 2**bits)[0]
    gt_16_value = nz_idx[gt_16_idx]

    filler = (gt_16_value -1)/(2**bits)
    cumsum_filler = np.cumsum(filler)
    
    tmp_nz_idx = np.zeros(nz_idx.size).astype(np.int32)
    tmp_nz_idx[le_16_idx] = le_16_idx
    tmp_nz_idx[gt_16_idx] = filler

    # append a element in case idx =-1
    cumsum_filler_tmp = np.append(cumsum_filler, [0])
    # ranging from [-1, len(gt_16_idx)-1]
    split = 10
    le_16_idx_adder_idx = np.zeros(le_16_idx.size).astype(np.int32)
    for i in range(split):
        le_16_idx_adder_idx[i::split] = np.sum( (le_16_idx[i::split].reshape(1, -1) > gt_16_idx.reshape(-1, 1)), axis = 0) -1
    le_16_idx_adder = cumsum_filler_tmp[le_16_idx_adder_idx]
    # le_16_idx_new = np.cumsum(tmp_nz_idx)[le_16_idx]
    le_16_idx_new = le_16_idx + le_16_idx_adder
    
    gt_16_idx_new = gt_16_idx + cumsum_filler
    gt_16_value_new = gt_16_value - filler*(2**bits)
    if flag == check_layer:
        print "le_16_idx: \n", le_16_idx
        print "le_16_value: \n", le_16_value
        print "le_16_idx_new: \n", le_16_idx_new
        
        print "gt_16_idx: \n", gt_16_idx
        print "gt_16_value: \n", gt_16_value
        print "gt_16_idx_new: \n", gt_16_idx_new
        print "gt_16_value_new: \n", gt_16_value_new


    # num_inserted = np.sum(cumsum_filler)
    if cumsum_filler.size ==0:
        num_inserted = 0
    else:
        num_inserted = cumsum_filler[-1]
    param_idx = np.zeros(len(nz_idx) + num_inserted).astype(np.int32)
    param_idx_valued = np.zeros(nz_idx.size).astype(np.int32)
    param_idx_valued[le_16_idx] = le_16_idx_new
    param_idx_valued[gt_16_idx] = gt_16_idx_new
    param_idx[:] = 2**bits
    param_idx[le_16_idx_new] = le_16_value
    param_idx[gt_16_idx_new] = gt_16_value_new
    
    
    '''
    # tobe store
    param_idx = np.array([]).astype(np.int64)   
    # actural valued idx in param_idx
    param_idx_valued = np.array([]).astype(np.int64)    
    num_inserted = 0

    for i, idx in enumerate(nz_idx):
        if idx <= 2**bits:
            # append current idx to param_idx
            param_idx = np.append(param_idx, idx)
            param_idx_valued = np.append(param_idx_valued, i+num_inserted)
        else:
            # insert some 2**bits to param_idx
            num_to_insert = (idx-1)// (2**bits)
            num_inserted += num_to_insert
            param_idx = np.concatenate((param_idx, np.array([2**bits]*num_to_insert), np.array([idx - 2**bits*num_to_insert])))
            param_idx_valued = np.append(param_idx_valued, i+num_inserted)

    # check length
    if len(param_idx) != len(nz_wb)+num_inserted:
        print "Error: length does not match (%d != %d)!"%(len(param_idx), len(nz_wb)+num_inserted)
        sys.exit()
    '''

    num_param = len(param_idx)
    wb_to_store = np.zeros(num_param)
    wb_to_store[param_idx_valued] = nz_wb
    max_exp = np.log2(np.max(np.fabs(nz_wb)))
    min_exp = max_exp - num_quantum_exp+1
    codebook = np.zeros(2**bits)

    #[0.0, -2^-7, -2^-6, ..., -2^-1, 2^-7, 2^-6, ..., 2^-1]
    for i in range(1, 2**(bits-1)): # 1 ~ 8
        # the last element in codebook is not used.
        codebook[i] = - np.exp2(min_exp + i -1)
        codebook[i+num_quantum_exp] = np.exp2(min_exp + i -1)
     
    if flag == check_layer:
        print "codebook:\n "
        pp.pprint(zip( map(hex, [i for i in range(len(codebook))]), [val for val in codebook]))
        print "num_param: \n", num_param
     
    value_to_bits = {}
    for i, val in enumerate(codebook):
        if i < len(codebook) -1:
            value_to_bits[val] = i

    vfunc = lambda x: value_to_bits[x]
    # in case num_param is odd
    if num_param %2 == 1:
        wb_to_store = np.append(wb_to_store, [0])
        param_idx = np.append(param_idx, [1])  

     
    if flag == check_layer:
        print "size of wb_to_store after append: \n", len(wb_to_store)
        print "wb_to_store: \n"
        pp.pprint(wb_to_store)
     
    param_idx = param_idx - 1
    if (param_idx >= 2**bits).any():
        print "Error: param_idx should not be great than or equal to %d", 2**bits
        sys.exit()
    # change int64 to uint8
    param_idx = param_idx.astype(np.uint8)
     
    if flag == check_layer:
        print "param_idx: \n", param_idx
        print "param_idx uint8:\n", list(map(hex, param_idx))
     
    wb_tmp = np.array(list(map(vfunc, wb_to_store))).astype(np.uint8)

     
    if flag == check_layer:
        print "wb_float to uint8:\n", list(map(hex,wb_tmp))
        print "wb_float to uint8:\n", wb_tmp
     
    wb_low_bit = wb_tmp[np.arange(0, len(wb_to_store), 2)]*2**bits + wb_tmp[np.arange(1, len(wb_to_store), 2)]
    # clip to [0~15]
    param_idx_low_bit = param_idx[np.arange(0, len(param_idx), 2)]* 2**bits + param_idx[np.arange(1, len(param_idx), 2)]

     
    if flag == check_layer:
        print "wb_lb : \n", list(map(hex,wb_low_bit))
        print "idx_lb: \n", list(map(hex, param_idx_low_bit))
     
    # return (num_params, codebook, wb, idx)
    return num_param, codebook.astype(np.float32), wb_low_bit.astype(np.uint8), param_idx_low_bit.astype(np.uint8)

def save_data(f_out, num, codebook, lb_wb, lb_idx):
    np.array([num], dtype = np.int32).tofile(f_out)
    codebook.tofile(f_out)
    lb_wb.tofile(f_out)
    lb_idx.tofile(f_out)
    

f_out = open(f_target_model, 'w')

total_start = time.time()
for i, param_name in enumerate(param_name_list):
    start = time.time()
    now = datetime.datetime.now()
    print "encoding layer [%s] weight ..."%param_name, now.strftime('%Y-%m-%d %H:%M:%S')
     
     
    if i == check_layer:
        print "====================layer no. [%d] saving [%s] weights ========================="%(i,param_name)
     
    w_inq = net.params[param_name][0].data.astype(np.float32).flatten()
    print "             num of weights: %d"%w_inq.size
    num_wb, codebook_wb, lb_wb, lb_idx = param_to_compact(w_inq, i)
    save_data(f_out, num_wb, codebook_wb, lb_wb, lb_idx)

    print "                %s  bias ..."%(" "*len(param_name))
     
    if i == check_layer:
        print "====================layer no. [%d] saving [%s] bias ========================"%(i, param_name)
     
    b_inq = net.params[param_name][1].data.astype(np.float32).flatten()
    print "             num of bias: %d"%b_inq.size
    num_wb, codebook_wb, lb_wb, lb_idx = param_to_compact(b_inq, i)
    save_data(f_out, num_wb, codebook_wb, lb_wb, lb_idx)
    end = time.time()
    print "time for layer [%s]: %f seconds"%(param_name, end-start )
 
total_end = time.time()
total_time = total_end -total_start
print ""
print "total time: %f s"%total_time
print ""

f_out.close()


