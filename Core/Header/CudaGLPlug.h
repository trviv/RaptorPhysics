#ifndef CUDA_GL_PLUG
#define CUDA_GL_PLUG

#include <cuda_gl_interop.h>
#include <cudaGL.h>
#include "CudaQuery.h"
#include "GLClass.h"

class CudaGLPlug
{
  cudaGraphicsResource_t  device_resource;

public:
  FORCE_INLINE CudaGLPlug()
  {
  }

  FORCE_INLINE CudaGLPlug(const GLObject& object)
  {
    setGLResource(object);
    CU_PROMPT;
  }

  FORCE_INLINE CU_HOST void setGLResource(const GLObject& object)
  {
    cudaGraphicsGLRegisterBuffer(&device_resource, object.get(), cudaGraphicsRegisterFlagsNone);
    CU_PROMPT;
  }

  CU_HOST void* map()
  {
    //map vertex buffer
    //take device memory pointers from open gl into the code
    //note this locks the open gl memory and open gl won't be able to access this unless freed
    cudaGraphicsMapResources(1, &device_resource);
    CU_PROMPT;
    void* device_buffer;
    size_t size;
    cudaGraphicsResourceGetMappedPointer((void**)&device_buffer, &size, device_resource);
    CU_PROMPT;
    return device_buffer;
  }

  CU_HOST void unmap()
  {
    cudaGraphicsUnmapResources(1, &device_resource);
  }
};

#endif