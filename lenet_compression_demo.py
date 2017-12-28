# this script assumes that you have the mnist train and test data in the "examples/mnist/" folder.
import os

dir_path = "lenet_compact_test"
if (not os.path.exists(dir_path)) or (not os.path.isdir(dir_path)):
    os.makedirs(dir_path)

sep_num = 40

# ref model
print "="*sep_num
print "Training the ref model ..."
os.system("nohup ./build/tools/caffe train -solver=examples/mnist/lenet_solver.prototxt -gpu=all > " + dir_path + "/lenet_ref.log 2>&1")

# DNS model
print "="*sep_num
print "Traing the DNS model ..."
os.system("nohup ./build/tools/caffe train -solver=examples/mnist/lenet_solver_DNS.prototxt -weights=examples/mnist/lenet_iter_10000.caffemodel -gpu=all > " + dir_path + "/lenet_dns.log 2>&1")

# convert DNS raw model to INQ raw model
print "="*sep_num
print "Converting DNS model to INQ model ..."
os.system("nohup python ./compression_scripts/model2INQ_raw.py examples/mnist/lenet_train_test_DNS.prototxt examples/mnist/lenet_DNS_iter_100000.caffemodel examples/mnist/lenet_train_test_inq_30.prototxt examples/mnist/inq_raw_ref.caffemodel > " + dir_path + "/lenet_dns_inq_convert.log 2>&1")

print "Lenet DNS pruning rate: "
os.system("tail -n 27 " + dir_path +"/lenet_dns_inq_convert.log")

# INQ 30% model
print "="*sep_num
print "Traing the INQ 30% model ..."
os.system("nohup ./build/tools/caffe train -solver=examples/mnist/lenet_solver_inq_30.prototxt -weights=examples/mnist/inq_raw_ref.caffemodel -gpu=all > " + dir_path + "/lenet_inq_30.log 2>&1")

# INQ 60% model
print "="*sep_num
print "Traing the INQ 60% model ..."
os.system("nohup ./build/tools/caffe train -solver=examples/mnist/lenet_solver_inq_60.prototxt -weights=examples/mnist/inq_30_iter_30000.caffemodel -gpu=all > " + dir_path + "/lenet_inq_60.log 2>&1")

# INQ 80% model
print "="*sep_num
print "Traing the INQ 80% model ..."
os.system("nohup ./build/tools/caffe train -solver=examples/mnist/lenet_solver_inq_80.prototxt -weights=examples/mnist/inq_60_iter_30000.caffemodel -gpu=all > " + dir_path + "/lenet_inq_80.log 2>&1")

# INQ 60% model
print "="*sep_num
print "Traing the INQ 90% model ..."
os.system("nohup ./build/tools/caffe train -solver=examples/mnist/lenet_solver_inq_90.prototxt -weights=examples/mnist/inq_80_iter_30000.caffemodel -gpu=all > " + dir_path + "/lenet_inq_90.log 2>&1")

# INQ 100% model, fully quantized model
print "="*sep_num
print "Traing the INQ 100% model (finally) ..."
os.system("nohup ./build/tools/caffe train -solver=examples/mnist/lenet_solver_inq_100.prototxt -weights=examples/mnist/inq_90_iter_30000.caffemodel -gpu=all > " + dir_path + "/lenet_inq_100.log 2>&1")

# Compress the final DNS+INQ model
print "="*sep_num
print "Compressing the model ... "
os.system("nohup python compression_scripts/encode_compact.py examples/mnist/lenet_train_test_inq_100.prototxt examples/mnist/inq_100_iter_1.caffemodel "+dir_path+ "/lenet_encoded.binary > " + dir_path + "/lenet_encoding.log 2>&1")
os.system("tail -n 30 "+dir_path +"/lenet_encoding.log")

# Decoding (recovering) the compressed model
print "="*sep_num
print "Decoding the compressed lenet model ..."
os.system("nohup python compression_scripts/decode_compact.py examples/mnist/lenet_train_test.prototxt lenet_encoded.binary "+ dir_path+"/lenet_decoded_normal.caffemodel > "+ dir_path +"/lenet_decoding.log 2>&1")
os.system("tail -n 16 "+dir_path+ "/lenet_decoding.log")



