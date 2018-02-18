#ifndef COMPUTE_HEADER_SHADER
#define COMPUTE_HEADER_SHADER

static uint threadIndex()
{
  return get_global_id(0) + get_global_id(1) * get_global_size(0);
}

static uint threadLocalIndex()
{
  return get_local_id(0) + get_local_id(1) * get_local_size(0);
}

static uint groupSize()
{
  return get_local_size(0) * get_local_size(1);
}

static uint groupIndex()
{
  return get_group_id(0);
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

uint mExpOf2(uint integer)
{
  if (!integer)
  {
    return 0;
  }

  int exp;
  uint backup = integer;
  for (exp = -1; integer; integer >>= 1)
  {
    exp++;
  }
  if ((((1 << exp) - 1) & backup)) exp++;
  return exp;
}

float sqr(const float x)
{
  return x*x;
}

ALIGN(4)
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

#endif