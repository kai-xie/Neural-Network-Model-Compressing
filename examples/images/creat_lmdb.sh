DATA=examples/images
rm -rf $DATA/img_test_lmdb
build/tools/convert_imageset -shuffle /home/users/xieqikai/myGitRepo/caffe-0828-compress/examples/images/ $DATA/test.txt $DATA/img_test_lmdb

