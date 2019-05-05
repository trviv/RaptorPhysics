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

static uint threadGroupSize()
{
  return get_local_size(0);
}

static uint threadGroupIndex()
{
  return get_group_id(0);
}

static uint threadGroupCount()
{
  return get_num_groups(0);
}

void globalMemBarrier()
{
  barrier(CLK_GLOBAL_MEM_FENCE);
}

void localMemBarrier()
{
  barrier(CLK_LOCAL_MEM_FENCE);
}

void localMemFence()
{
  mem_fence(CLK_LOCAL_MEM_FENCE);
}

#define Kernel  __kernel
#define Device  __global
#define Const   __constant
#define Shared  __local
#define Thread  __private

#define COMPUTE_SHADER_SCOPE
#define COMPUTE_MAX_THREADS     1024
#define COMPUTE_SUB_GROUP_SIZE  32
#define COMPUTE_SUB_GROUP_EXP   5
#define COMPUTE_EPSILON         0.000001f

#define ALIGN(n)              __attribute__((aligned(n))) __attribute__((packed))
#define DEFAULT_ALIGN         ALIGN(16)

#define constructFloat2       (float2)
#define constructFloat3       (float3)
#define constructUint3        (uint3)
#define constructUint2        (uint2)
#define constructInt2         (int2)

#define NUM_BANKS       16
#define LOG_NUM_BANKS   4

inline const uint paddedIndex(const uint n)
{
  return n;// +(((n >> NUM_BANKS) + n) >> (LOG_NUM_BANKS << 1));
}

float sqr(const float x)
{
  return x * x;
}

typedef struct
{
  float val[9];
} Matrix3x3;

#define addMatrix3x3(a, b)    { for (uint i = 0; i < 9; i++) { (a)->val[i] += (b)->val[i]; } }
#define divMatrix3x3(a, b)    { for (uint i = 0; i < 9; i++) { (a)->val[i] /= (*b); } }
#define copyMatrix3x3(a, b)   { for (uint i = 0; i < 9; i++) { (a)->val[i] = (b)->val[i]; } }
#define clearMatrix3x3(a, b)  { for (uint i = 0; i < 9; i++) { (a)->val[i] = b; } }

#define atomicLoad(location)        atomic_add (location, 0)
#define atomicSave(location, value) atomic_xchg(location, value)
#define atomicAdd(location, value)  atomic_add (location, value)

#endif