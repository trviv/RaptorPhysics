#ifndef COMPUTE_HEADER_SHADER
#define COMPUTE_HEADER_SHADER

static uint threadIndex()
{
  return get_global_id(0) + get_global_id(1)*get_global_size(0);
}

#define Kernel  __kernel
#define Device  __global
#define Const   __constant
#define Group   __local
#define Thread  __private

#define COMPUTE_SHADER_SCOPE

#define ALIGN(n) __attribute__((aligned(n))) __attribute__((packed))
#define DEFAULT_ALIGN ALIGN(16) 

#define makeFloat3 (float3)

#define COMPUTE_EPSILON .000001f

#endif