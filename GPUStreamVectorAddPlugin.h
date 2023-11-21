#ifndef GPUSTREAMVECTORADDPLUGIN_H
#define GPUSTREAMVECTORADDPLUGIN_H

#include "Plugin.h"
#include "Tool.h"
#include "PluginProxy.h"
#include <string>
#include <map>

class GPUStreamVectorAddPlugin : public Plugin, public Tool {

	public:
		void input(std::string file);
		void run();
		void output(std::string file);
	private:
                std::string inputfile;
		std::string outputfile;
 //               std::map<std::string, std::string> parameters;
};

// Compute C = A * B
// Sgemm stands for single precision general matrix-matrix multiply
__global__ void sgemm(float *A, float *B, float *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns) {
  //@@ Insert code to implement matrix multiplication here
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < numARows && col < numBColumns) {
    float sum = 0;
    for (int ii = 0; ii < numAColumns; ii++) {
      sum += A[row * numAColumns + ii] * B[ii * numBColumns + col];
    }
    C[row * numBColumns + col] = sum;
  }
}
__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  //@@ Insert code to implement vector addition here
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < len)
    out[i] = in1[i] + in2[i];
}

#endif
