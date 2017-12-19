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
    f_target_net = sys.argv[2]
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


def param_to_compact(wb):
    nz_idx = np.nonzero(wb)
    np.diff(nz_idx)





for param_name in param_name_list:
    if len(net.params[param_name]) == 4:
        w_inq = net.params[param_name][0].data.astype(np.float32).flatten()
        nz_inx_w = param_to_compact(w_inq)
        b_inq = net.params[param_name][1].data.astype(np.float32).flatten()


        w_dns_mask = net.params[param_name][2].data.astype(np.float32).flatten()
        b_dns_mask = net.params[param_name][3].data.astype(np.float32).flatten()    

        total_params, params_kept = display_layer_info(param_name, w_dns_mask, b_dns_mask, total_params, params_kept ) 
        dns_to_target(net_target.params[param_name][0].data, w_dns, w_dns_mask)
        dns_to_target(net_target.params[param_name][1].data, b_dns, b_dns_mask)
    elif len(net.params[param_name]) == 2:
        w_src = net.params[param_name][0].data.astype(np.float32).flatten()
        b_src = net.params[param_name][1].data.astype(np.float32).flatten()
        normal_to_target(net_target.params[param_name][0].data, w_src)
        normal_to_target(net_target.params[param_name][1].data, b_src)

    else:
        print "Error: len of net.params[%s] is %d"%(param_name, len((net.params[param_name])) )




