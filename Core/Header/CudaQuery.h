#ifndef RX_CUDA_QUERY
#define RX_CUDA_QUERY

#include "Root.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_texture_types.h>

static const char *_cudaGetErrorEnum(cudaError_t error)
{
  return cudaGetErrorString(error);
}

#define DEVICE_RESET cudaDeviceReset();

template< typename T > void check
(T result, char const *const func, const char *const file, int const line)
{
  if (result)
  {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
      file, line,
      static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    DEVICE_RESET
      // Make sure we call CUDA Device Reset before exiting
      exit(EXIT_FAILURE);
  }
}
#define checkCudaErrors(val)  check ( (val), #val, __FILE__, __LINE__ )

#endif