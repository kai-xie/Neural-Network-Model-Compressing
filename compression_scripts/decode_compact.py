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

decode compact binary model to normal caffemodel

Usage:
    decode_compact.py <src_net.prototxt> <compact_model.binary> <normal_model.caffemodel>

"""

if len(sys.argv) != 4:
    print help_
    sys.exit(-1)
else:
    f_src_net = sys.argv[1]
    f_bin_model = sys.argv[2]
    f_normal_model = sys.argv[3]

if not os.path.exists(f_src_net):
    print "Error: %s does not exist!"%(f_src_net)
    sys.exit()
elif not os.path.exists(f_bin_model):
    print "Error: %s does not exist!"%(f_bin_model)
    sys.exit()
'''
elif not os.path.exists(f_normal_model):
    print "Error: %s does not exist!"%(f_normal_model)
    sys.exit()
'''

caffe.set_mode_cpu()

net = caffe.Net(f_src_net, caffe.TEST)
layers = filter(lambda x:'conv' in x or 'fc' in x or 'ip' in x, net.params.keys())

f_in = open(f_bin_model, 'rb')
bits = 4
num_quantum_value = 7

def decode_data(net_data, wb_lb, idx_lb, codebook, num_nz_wb):
    data = np.zeros(net_data.size)
    if num_nz_wb == 0:
        data = data.reshape(net_data.shape)
        np.copyto(net_data, data)
        return

    num_tmp = ((num_nz_wb -1)/2 +1 )*2
    nz_wb = np.zeros(num_tmp, dtype = np.uint8)
    nz_idx = np.zeros(num_tmp, dtype = np.uint8)

    nz_wb[np.arange(0, num_tmp, 2)] = wb_lb / (2**bits)
    nz_wb[np.arange(1, num_tmp, 2)] = wb_lb % (2**bits)
    nz_idx[np.arange(0, num_tmp, 2)] = idx_lb / (2**bits)
    nz_idx[np.arange(1, num_tmp, 2)] = idx_lb % (2**bits)
 
    nz_wb = nz_wb[np.arange(num_nz_wb)]
    nz_idx = nz_idx[np.arange(num_nz_wb)]

    # Recover the matrix
    nz_idx = np.cumsum(nz_idx+1) -1
    code = np.zeros(net_data.size, dtype = np.uint8)
    code[nz_idx] = nz_wb
    data = np.reshape(codebook[code], net_data.shape)
    np.copyto(net_data, data)
    

for layer in layers:
    ### Weights ###
    num_nz_wb = np.fromfile(f_in, dtype = np.int32, count = 1)  # num of non-zero weight/bias
    codebook_size = 2**bits
    codebook = np.fromfile(f_in, dtype = np.float32, count = codebook_size)
    lb_count = (num_nz_wb - 1)/2 +1
    # low bit weight, assuming stored in 4 bits
    wb_lb = np.fromfile(f_in, dtype = np.uint8, count = lb_count)
    # low bit index, 4 bits
    idx_lb = np.fromfile(f_in, dtype = np.uint8, count = lb_count)
    decode_data(net.params[layer][0].data, wb_lb, idx_lb, codebook, num_nz_wb)

    ### Bias ###
    num_nz_wb = np.fromfile(f_in, dtype = np.int32, count = 1)  # num of non-zero weight/bias
    codebook_size = 2**bits
    codebook = np.fromfile(f_in, dtype = np.float32, count = codebook_size)
    lb_count = (num_nz_wb - 1)/2 +1
    # low bit weight, assuming stored in 4 bits
    wb_lb = np.fromfile(f_in, dtype = np.uint8, count = lb_count)
    # low bit index, 4 bits
    idx_lb = np.fromfile(f_in, dtype = np.uint8, count = lb_count)
    decode_data(net.params[layer][1].data, wb_lb, idx_lb, codebook, num_nz_wb)
    
net.save(f_normal_model)
f_in.close()

print ""
print "Model decoded. Saved as %s"%f_normal_model
print ""




