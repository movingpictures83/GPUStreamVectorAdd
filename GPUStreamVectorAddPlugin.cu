 

#include "GPUStreamVectorAddPlugin.h"

void GPUStreamVectorAddPlugin::input(std::string infile) {
  readParameterFile(infile);
}

void GPUStreamVectorAddPlugin::run() {}

void GPUStreamVectorAddPlugin::output(std::string outfile) {
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  cudaStream_t stream[4];
  float *d_A[4], *d_B[4], *d_C[4];
  int i, k, Seglen = 1024;
  int Gridlen = (Seglen - 1) / 256 + 1;

 inputLength = atoi(myParameters["N"].c_str());
 hostInput1 = (float*) malloc(inputLength*sizeof(float));
 hostInput2 = (float*) malloc(inputLength*sizeof(float));
 std::ifstream myinput((std::string(PluginManager::prefix())+myParameters["vector1"]).c_str(), std::ios::in);
 for (i = 0; i < inputLength; ++i) {
        float k;
        myinput >> k;
        hostInput1[i] = k;
 }
 std::ifstream myinput2((std::string(PluginManager::prefix())+myParameters["vector2"]).c_str(), std::ios::in);
 for (i = 0; i < inputLength; ++i) {
        float k;
        myinput2 >> k;
        hostInput2[i] = k;
 }


  hostOutput = (float *)malloc((4*Seglen+inputLength) * sizeof(float));
  
  
  for (i = 0; i < 4; i++) {
    cudaStreamCreate(&stream[i]);
    cudaMalloc((void **)&d_A[i], (4*Seglen+inputLength) * sizeof(float));
    cudaMalloc((void **)&d_B[i], (4*Seglen+inputLength) * sizeof(float));
    cudaMalloc((void **)&d_C[i], (4*Seglen+inputLength) * sizeof(float));
  }

  for (i = 0; i < inputLength; i += Seglen * 4) {
    for (k = 0; k < 4; k++) {
      cudaMemcpyAsync(d_A[k], hostInput1 + i + k * Seglen,
                      Seglen * sizeof(float), cudaMemcpyHostToDevice,
                      stream[k]);
      cudaMemcpyAsync(d_B[k], hostInput2 + i + k * Seglen,
                      Seglen * sizeof(float), cudaMemcpyHostToDevice,
                      stream[k]);
      vecAdd<<<Gridlen, 256, 0, stream[k]>>>(d_A[k], d_B[k], d_C[k],
                                             Seglen);
    }
    cudaStreamSynchronize(stream[0]);
    cudaStreamSynchronize(stream[1]);
    cudaStreamSynchronize(stream[2]);
    cudaStreamSynchronize(stream[3]);
    for (k = 0; k < 4; k++) {
      cudaMemcpyAsync(hostOutput + i + k * Seglen, d_C[k],
                      Seglen * sizeof(float), cudaMemcpyDeviceToHost,
                      stream[k]);
    }
  }
  cudaDeviceSynchronize();
 std::ofstream outsfile(outfile.c_str(), std::ios::out);
        for (i = 0; i < inputLength; ++i){
                outsfile << hostOutput[i];//std::setprecision(0) << a[i*N+j];
                outsfile << "\n";
        }


  free(hostInput1);
  free(hostInput2);
  free(hostOutput);
  
  cudaStreamDestroy(stream[0]);
  cudaStreamDestroy(stream[1]);
  cudaStreamDestroy(stream[2]);
  cudaStreamDestroy(stream[3]);

  for (k = 0; k < 4; k++) {
    cudaFree(d_A[k]);
    cudaFree(d_B[k]);
    cudaFree(d_C[k]);
  }

}


PluginProxy<GPUStreamVectorAddPlugin> GPUStreamVectorAddPluginProxy = PluginProxy<GPUStreamVectorAddPlugin>("GPUStreamVectorAdd", PluginManager::getInstance());

