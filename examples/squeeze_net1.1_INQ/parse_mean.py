'''
convert binary caffemodel to human-readable text file.
'''

import os, sys
import numpy as np

caffe_root = os.environ["CAFFE_ROOT"]
sys.path.append(caffe_root+"/python")
sys.path.append(caffe_root+"/python/caffe/proto")
import caffe

help = '''
Usage:
    parse_mean.py <mean.binary> <output.txt>
'''

if len(sys.argv) != 3:
    print help
    sys.exit(-1)
else:
    binary_model = sys.argv[1]  #
    output = sys.argv[2]   #

if not os.path.exists(binary_model):
    print "Error: %s does not exist!"%(binary_model)
    sys.exit()


model = caffe.proto.caffe_pb2.BlobProto()
f = open(binary_model, 'rb')
print "Parsing %s ..."%binary_model
model.ParseFromString(f.read())
f.close()

f = open(output, 'w')
# print 'model type: ', type(model)
print "Saving file: %s ..."%(output)
print >> f,model
# print >> f,model.__str__
f.close()

print "Success: file saved as %s"%(output)

mean_array = np.array(caffe.io.blobproto_to_array(model))
print mean_array
print mean_array.shape
print mean_array[0].shape
print mean_array[0]

m1 = np.mean(mean_array[0][0])
m2 = np.mean(mean_array[0][1])
m3 = np.mean(mean_array[0][2])
print "mean values: ",m1, m2, m3
print "mean values: %d, %d, %d "%(np.round(m1),np.round(m2),np.round(m3))




