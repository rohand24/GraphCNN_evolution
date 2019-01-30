

TF_INC=/usr/local/lib/python2.7/dist-packages/tensorflow/include
#TF_LIB=/usr/local/lib/python2.7/dist-packages/tensorflow

gcc -std=c++11 -shared sparse_tensor_sparse_matmul.cc -o sparse_tensor_sparse_matmul.so  -fPIC -I$TF_INC -O2
#-I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -O2   



