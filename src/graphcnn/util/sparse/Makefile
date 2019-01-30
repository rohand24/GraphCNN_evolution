TF_INC = `python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())'`
TF_LIB = `python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())'`
TF_CFLAGS=`python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))'`
CXX = g++
VER = -std=c++11
CFLAGS = -g -fPIC -I$(TF_INC) -I$(TF_INC)/external/nsync/public -L$(TF_LIB) -ltensorflow_framework -D_GLIBCXX_USE_CXX11_ABI=0
GPUCC = nvcc
GPUCFLAGS = -c
GPULFLAGS = -x cu -Xcompiler -fPIC
GPUDEF = -D GOOGLE_CUDA=1

default: gpu

cpu:
	$(CXX) $(VER) -shared SparseConv.cc -o SparseConv.so $(CFLAGS)
	$(CXX) $(VER) -shared SparseConvGrad.cc -o SparseConvGrad.so $(CFLAGS)
	$(CXX) $(VER) -shared SparseAverageVertexPool.cc -o SparseAverageVertexPool.so $(CFLAGS)
	$(CXX) $(VER) -shared SparseAverageVertexPoolGrad.cc -o SparseAverageVertexPoolGrad.so $(CFLAGS)
	$(CXX) $(VER) -shared SparseMaxVertexPool.cc -o SparseMaxVertexPool.so $(CFLAGS)
	$(CXX) $(VER) -shared SparseMaxVertexPoolGrad.cc -o SparseMaxVertexPoolGrad.so $(CFLAGS)
	$(CXX) -std=c++11 -shared SMMPUtils.cc SparseSparseMatMulControl.cc SparseSparseBatchMatMul.cc -o SparseSparseBatchMatMul.so $(CFLAGS)
	$(CXX)  -std=c++11 -shared SMMPUtils.cc SparseSparseMatMulControl.cc SparseSparseBatchMatMulGrad.cc -o SparseSparseBatchMatMulGrad.so $(CFLAGS)

gpu: ConstOut_cu.o \
	 SparseConv_cu.o \
	 SparseConvGrad_cu.o \
	 SparseAverageVertexPool_cu.o \
	 SparseAverageVertexPoolGrad_cu.o \
	 SparseMaxVertexPool_cu.o \
	 SparseMaxVertexPoolGrad_cu.o \
	 SparseEdgeConv_cu.o \
	 SparseEdgeConvGrad_cu.o \
	 SparseConv.so \
	 SparseConvGrad.so \
	 SparseAverageVertexPool.so \
	 SparseAverageVertexPoolGrad.so \
	 SparseMaxVertexPool.so \
	 SparseMaxVertexPoolGrad.so \
	 SparseSparseBatchMatMul.so \
	 SparseSparseBatchMatMulGrad.so \
	 SparseEdgeConv.so \
	 SparseEdgeConvGrad.so \

ConstOut_cu.o: ConstOut.cu.cc
	$(GPUCC) -std=c++11 -c -o ConstOut_cu.o ConstOut.cu.cc -I/usr/local $(TF_CFLAGS) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -O2
SparseConv_cu.o: SparseConv.cu.cc
	$(GPUCC) -std=c++11 -c -o SparseConv_cu.o SparseConv.cu.cc -I/usr/local $(TF_CFLAGS) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -O2
SparseConvGrad_cu.o: SparseConvGrad.cu.cc
	$(GPUCC) -std=c++11 -c -o SparseConvGrad_cu.o SparseConvGrad.cu.cc -I/usr/local $(TF_CFLAGS) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -O2
SparseAverageVertexPool_cu.o: SparseAverageVertexPool.cu.cc
	$(GPUCC) -std=c++11 -c -o SparseAverageVertexPool_cu.o SparseAverageVertexPool.cu.cc -I/usr/local $(TF_CFLAGS) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -O2
SparseAverageVertexPoolGrad_cu.o: SparseAverageVertexPoolGrad.cu.cc
	$(GPUCC) -std=c++11 -c -o SparseAverageVertexPoolGrad_cu.o SparseAverageVertexPoolGrad.cu.cc -I/usr/local $(TF_CFLAGS) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -O2
SparseMaxVertexPool_cu.o: SparseMaxVertexPool.cu.cc
	$(GPUCC) -std=c++11 -c -o SparseMaxVertexPool_cu.o SparseMaxVertexPool.cu.cc -I/usr/local $(TF_CFLAGS) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -O2
SparseMaxVertexPoolGrad_cu.o: SparseMaxVertexPoolGrad.cu.cc
	$(GPUCC) -std=c++11 -c -o SparseMaxVertexPoolGrad_cu.o SparseMaxVertexPoolGrad.cu.cc -I/usr/local $(TF_CFLAGS) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -O2
SparseEdgeConv_cu.o: SparseEdgeConv.cu.cc
	$(GPUCC) -std=c++11 -c -o SparseEdgeConv_cu.o SparseEdgeConv.cu.cc -I/usr/local $(TF_CFLAGS) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -O2
SparseEdgeConvGrad_cu.o: SparseEdgeConvGrad.cu.cc
	$(GPUCC) -std=c++11 -c -o SparseEdgeConvGrad_cu.o SparseEdgeConvGrad.cu.cc -I/usr/local $(TF_CFLAGS) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -O2
SparseConv.so: ConstOut_cu.o SparseConv_cu.o SparseConv.cc
	$(CXX) $(VER) -shared ConstOut_cu.o SparseConv_cu.o SparseConv.cc  -o SparseConv.so -I/usr/local $(CFLAGS) -lcuda -DGOOGLE_CUDA=1
SparseConvGrad.so: ConstOut_cu.o SparseConvGrad_cu.o SparseConvGrad.cc
	$(CXX) $(VER) -shared ConstOut_cu.o SparseConvGrad_cu.o SparseConvGrad.cc  -o SparseConvGrad.so -I/usr/local $(CFLAGS) -lcuda -DGOOGLE_CUDA=1
SparseAverageVertexPool.so: SparseAverageVertexPool_cu.o SparseAverageVertexPool.cc
	$(CXX) $(VER) -shared ConstOut_cu.o SparseAverageVertexPool_cu.o SparseAverageVertexPool.cc  -o SparseAverageVertexPool.so -I/usr/local $(CFLAGS) -lcuda -DGOOGLE_CUDA=1
SparseAverageVertexPoolGrad.so: SparseAverageVertexPoolGrad_cu.o SparseAverageVertexPoolGrad.cc
	$(CXX) $(VER) -shared ConstOut_cu.o SparseAverageVertexPoolGrad_cu.o SparseAverageVertexPoolGrad.cc  -o SparseAverageVertexPoolGrad.so -I/usr/local $(CFLAGS) -lcuda -DGOOGLE_CUDA=1
SparseMaxVertexPool.so: ConstOut_cu.o SparseMaxVertexPool_cu.o SparseMaxVertexPool.cc
	$(CXX) $(VER) -shared ConstOut_cu.o SparseMaxVertexPool_cu.o SparseMaxVertexPool.cc  -o SparseMaxVertexPool.so -I/usr/local $(CFLAGS) -lcuda -DGOOGLE_CUDA=1
SparseMaxVertexPoolGrad.so: ConstOut_cu.o SparseMaxVertexPoolGrad_cu.o SparseMaxVertexPoolGrad.cc
	$(CXX) $(VER) -shared ConstOut_cu.o SparseMaxVertexPoolGrad_cu.o SparseMaxVertexPoolGrad.cc  -o SparseMaxVertexPoolGrad.so -I/usr/local $(CFLAGS) -lcuda -DGOOGLE_CUDA=1
SparseSparseBatchMatMul.so: SMMPUtils.cc SparseSparseMatMulControl.cc SparseSparseBatchMatMul.cc
	$(CXX) -std=c++11 -shared SMMPUtils.cc SparseSparseMatMulControl.cc SparseSparseBatchMatMul.cc -o SparseSparseBatchMatMul.so $(CFLAGS)
SparseSparseBatchMatMulGrad.so: SMMPUtils.cc SparseSparseMatMulControl.cc SparseSparseBatchMatMulGrad.cc
	$(CXX)  -std=c++11 -shared SMMPUtils.cc SparseSparseMatMulControl.cc SparseSparseBatchMatMulGrad.cc -o SparseSparseBatchMatMulGrad.so $(CFLAGS)
SparseEdgeConv.so: ConstOut_cu.o SparseEdgeConv_cu.o SparseEdgeConv.cc
	$(CXX) $(VER) -shared ConstOut_cu.o SparseEdgeConv_cu.o SparseEdgeConv.cc  -o SparseEdgeConv.so -I/usr/local $(CFLAGS)  -lcuda -DGOOGLE_CUDA=1
SparseEdgeConvGrad.so: ConstOut_cu.o SparseEdgeConvGrad_cu.o SparseEdgeConvGrad.cc
	$(CXX) $(VER) -shared ConstOut_cu.o SparseEdgeConvGrad_cu.o SparseEdgeConvGrad.cc  -o SparseEdgeConvGrad.so -I/usr/local $(CFLAGS) -lcuda -DGOOGLE_CUDA=1
clean:
	rm -f *.so
	rm -f *.o