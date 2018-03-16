#ifndef COMPUTE_HEADER_SHADER
#define COMPUTE_HEADER_SHADER

static uint threadIndex()
{
  return get_global_id(0);
}

static uint threadLocalIndex()
{
  return get_local_id(0);
}

static uint groupSize()
{
  return get_local_size(0);
}

static uint groupIndex()
{
  return get_group_id(0);
}

void globalMemBarrier()
{
  barrier(CLK_GLOBAL_MEM_FENCE);
}

void localMemBarrier()
{
  barrier(CLK_LOCAL_MEM_FENCE);
}

#define Kernel  __kernel
#define Device  __global
#define Const   __constant
#define Shared  __local
#define Thread  __private

#define COMPUTE_SHADER_SCOPE
#define COMPUTE_MAX_THREADS   1024
#define COMPUTE_EPSILON       0.000001f

#define ALIGN(n)              __attribute__((aligned(n))) __attribute__((packed))
#define DEFAULT_ALIGN         ALIGN(16)

#define constructFloat3       (float3)

float sqr(const float x)
{
  return x*x;
}

typedef struct
{
  float val[9];
} Matrix3x3;

inline void addMatrix3x3(Device Matrix3x3* a, const Device Matrix3x3* b)
{
  for (uint i = 0; i < 9; i++)
  {
    a->val[i] += b->val[i];
  }
}

inline void divMatrix3x3(Device Matrix3x3* a, const float* b)
{
  for (uint i = 0; i < 9; i++)
  {
    a->val[i] /= (*b);
  }
}

inline void copyMatrix3x3(Device Matrix3x3* a, const Device Matrix3x3* b)
{
  for (uint i = 0; i < 9; i++)
  {
    a->val[i] = b->val[i];
  }
}

#endif