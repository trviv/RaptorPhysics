#ifndef COMPUTE_HEADER_SHADER
#define COMPUTE_HEADER_SHADER

#ifdef USE_OPENCL_COMPUTE

#define constantKernelInput(type, variableName) const type variableName
#define atomicKernelInput(type, variableName) Device type *variableName
#define sharedMemKernelInput(type, variableName, index) Shared type *variableName

#define threadIndex()       get_global_id(0)
#define threadLocalIndex()  get_local_id(0)
#define threadGroupSize()   get_local_size(0)
#define threadGroupIndex()  get_group_id(0)
#define threadGroupCount()  get_num_groups(0)
#define threadIndexN(dim)       get_global_id(dim)
#define threadLocalIndexN(dim)  get_local_id(dim)
#define threadGroupSizeN(dim)   get_local_size(dim)
#define threadGroupIndexN(dim)  get_group_id(dim)
#define threadGroupCountN(dim)  get_num_groups(dim)
#define globalMemBarrier()  barrier(CLK_GLOBAL_MEM_FENCE)
#define localMemBarrier()   barrier(CLK_LOCAL_MEM_FENCE)
#define localMemFence()     mem_fence(CLK_LOCAL_MEM_FENCE)

#define Kernel  __kernel
#define Device  __global
#define Const   __constant
#define Shared  __local
#define Thread  __private

#define constructUshort4    (ushort4)
#define constructShort3     (short3)
#define constructShort2     (short2)
#define constructFloat2     (float2)
#define constructFloat3     (float3)
#define constructFloat4     (float4)
#define constructColor3     (colorType3)
#define constructColor4     (colorType4)
#define constructUint2      (uint2)
#define constructUint3      (uint3)
#define constructUint4      (uint4)
#define constructInt2       (int2)
#define constructInt3       (int3)
#define constructUchar3     (uchar3)
#define constructUchar4     (uchar4)

#define convertShort2(a)    convert_short2(a)
#define convertUshort4(a)   convert_ushort4(a)
#define convertInt3(a)      convert_int3(a)
#define asUchar4(x)         as_uchar4(x)
#define asFloat(x)          as_float(x)
#define asUint(x)           as_uint(x)
#define asUint3(x)          as_uint3(x)
#define asUshort(x)         as_ushort(x)
#define asUshort2(x)        as_ushort2(x)
#define simdAll(x)          assert
#define simdFirst(x)        assert
#define simdIsFirst()       assert
#define simdReduce(x)       assert
#define simdScan(x)         assert
#define simdMin(x)          assert
#define simdMax(x)          assert
#define bool3               int3
#define selectInput2(x)     (int2)(x)
#define selectInput3(x)     (int3)(x)
#define lengthSq(x)         dot(x, x)
#define min3(v0, v1, v2)    min(min(v0, v1), v2)
#define max3(v0, v1, v2)    max(max(v0, v1), v2)
#define minCompFloat3(vec)  min(min(vec.x, vec.y), vec.z)
#define maxCompFloat3(vec)  max(max(vec.x, vec.y), vec.z)
#define mulVecMatrix(vec, mat) (constructFloat4(dot(vec, mat.lo.lo), dot(vec, mat.lo.hi), dot(vec, mat.hi.lo), dot(vec, mat.hi.hi)))
#define mulMatrixVec(mat, vec) (constructFloat4(dot(vec, mat.s048c), dot(vec, mat.s159d), dot(vec, mat.s26ae), dot(vec, mat.s37bf)))
#define reflectVector(incident, normal) (incident – 2.f * dot(normal, incident) * normal)
inline float3 refractVector(const float3 incident, float3 normal, float eta)
{
  float dotNI = dot(normal, incident);
  if (dotNI < 0.f)
  {
    dotNI = -dotNI;
  }
  else
  {
    eta = 1.f/eta;
    normal = -normal;
  }
  const float k = 1.f - (eta * eta * (1.f - dotNI * dotNI));
  return select(0.f, normalize(incident * eta + normal * (eta * dotNI - sqrt(k))), k > 0.f);
}

#define atomicLoad(location)          atomic_or  ((Device uint*)location, 0)
#define atomicStore(location, value)  atomic_xchg((Device uint*)location, value)
#define atomicAdd(location, value)    atomic_add (location, value)
#define atomicMax(location, value)    atomic_max (location, value)
#define atomicMin(location, value)    atomic_min (location, value)
#define atomicCmpXchg(location, existingValue, desiredValue) \
  ((existingValue == atomic_cmpxchg((Device uint*)location, asUint(existingValue), asUint(desiredValue))) || (existingValue = atomicLoad(location) | true))

#define atomicLoadShared(location)        atomic_or((Shared uint*)location, 0)
#define atomicAddShared(location, value)  atomic_add((Shared uint*)location, value)
#define atomicAddSignedShared(location, value)  atomic_add((Shared int*)location, value)
#define atomicCmpXchgShared(location, existingValue, desiredValue) \
  ((existingValue == atomic_cmpxchg((Shared uint*)location, asUint(existingValue), asUint(desiredValue))) || (existingValue = atomicLoadShared(location) | true))

#define KERNEL_GLOBAL_ARGUMENTS
#define KERNEL_THREAD_ARGUMENTS
#define KERNEL_THREADGROUP_ARGUMENTS

#define ALIGN(n)            __attribute__((aligned(n))) __attribute__((packed))

#define commonInt8    int8
#define commonUint8   uint8
#define commonFloat8  float8

#define commonInt16   int16
#define commonUint16  uint16
#define commonFloat16 float16

#define float4x4      float16

#else

#define ASSUME_FLEXIBLE_VECTOR_ALIGNMENT

#define constantKernelInput(type, variableName) Const type& variableName
#define atomicKernelInput(type, variableName) Device atomic_##type *variableName
#define sharedMemKernelInput(type, variableName, index) Shared type *variableName[[ threadgroup(index) ]]

#define threadIndex()       thread_position_in_grid[0]
#define threadLocalIndex()  thread_index_in_threadgroup
#define threadGroupSize()   threads_per_threadgroup[0]
#define threadGroupIndex()  threadgroup_position_in_grid[0]
#define threadGroupCount()  threadgroups_per_grid[0]
#define threadIndexN(dim)       thread_position_in_grid[dim]
#define threadLocalIndexN(dim)  thread_position_in_threadgroup[dim]
#define threadGroupSizeN(dim)   threads_per_threadgroup[dim]
#define threadGroupIndexN(dim)  threadgroup_position_in_grid[dim]
#define threadGroupCountN(dim)  threadgroups_per_grid[dim]
#define globalMemBarrier()  threadgroup_barrier(mem_flags::mem_device)
#define localMemBarrier()   threadgroup_barrier(mem_flags::mem_threadgroup)
#define localMemFence()     threadgroup_barrier(mem_flags::mem_none)

#define Kernel  kernel
#define Device  device
#define Const   constant
#define Shared  threadgroup
#define Thread  thread

#define constructUshort4    ushort4
#define constructShort3     short3
#define constructShort2     short2
#define constructFloat2     float2
#define constructFloat3     float3
#define constructFloat4     float4
#define constructColor3     colorType3
#define constructColor4     colorType4
#define constructUint2      uint2
#define constructUint3      uint3
#define constructUint4      uint4
#define constructInt2       int2
#define constructInt3       int3
#define constructUchar3     uchar3
#define constructUchar4     uchar4

#define convertShort2(a)    short2(a)
#define convertUshort4(a)   ushort4(a)
#define convertInt3(a)      int3(a)
#define asUchar4(x)         as_type<uchar4>(x)
#define asFloat(x)          as_type<float>(x)
#define asUint(x)           as_type<uint>(x)
#define asUint3(x)          as_type<uint3>(x)
#define asUshort(x)         as_type<ushort>(x)
#define asUshort2(x)        as_type<ushort2>(x)
#define simdAll(x)          simd_all(x)
#define simdFirst(x)        simd_broadcast_first(x)
#define simdIsFirst()       simd_is_first()
#define simdReduce(x)       simd_sum(x)
#define simdScan(x)         simd_prefix_inclusive_sum(x)
#define simdMin(x)          simd_min(x)
#define simdMax(x)          simd_max(x)
#define selectInput2(x)     bool2(x)
#define selectInput3(x)     bool3(x)
#define lengthSq(x)         length_squared(x)
#define minCompFloat3(vec)  min3(vec.x, vec.y, vec.z)
#define maxCompFloat3(vec)  max3(vec.x, vec.y, vec.z)
#define mulVecMatrix(vec, mat) (vec * mat)
#define mulMatrixVec(mat, vec) (mat * vec)
#define reflectVector(incident, normal) reflect(incident, normal)
//#define refractVector(incident, normal, eta) refract(incident, normal, eta)
inline float3 refractVector(const float3 incident, float3 normal, float eta)
{
  float dotNI = dot(normal, incident);
  if (dotNI < 0.f)
  {
    dotNI = -dotNI;
  }
  else
  {
    eta = 1.f/eta;
    normal = -normal;
  }
  const float k = 1.f - (eta * eta * (1.f - dotNI * dotNI));
  return select(0.f, normalize(incident * eta + normal * (eta * dotNI - sqrt(k))), k > 0.f);
}

#define atomicLoad(location)          atomic_fetch_or_explicit((Device atomic_uint*)location, 0, memory_order_relaxed)
#define atomicStore(location, value)  atomic_exchange_explicit((Device atomic_uint*)location, value, memory_order_relaxed)
#define atomicAdd(location, value)    atomic_fetch_add_explicit((Device atomic_uint*)location, value, memory_order_relaxed)
#define atomicMax(location, value)    atomic_fetch_max_explicit((Shared atomic_uint*)location, value, memory_order_relaxed)
#define atomicMin(location, value)    atomic_fetch_min_explicit((Shared atomic_uint*)location, value, memory_order_relaxed)
#define atomicCmpXchg(location, existingValue, desiredValue) \
  atomic_compare_exchange_weak_explicit((Device atomic_uint*)location, ((Thread uint*)&existingValue), asUint(desiredValue), memory_order_relaxed, memory_order_relaxed)

#define atomicLoadShared(location)        atomic_fetch_or_explicit((volatile Shared atomic_uint*)location, 0, memory_order_relaxed)
#define atomicAddShared(location, value)  atomic_fetch_add_explicit((volatile Shared atomic_uint*)location, value, memory_order_relaxed)
#define atomicAddSignedShared(location, value)  atomic_fetch_add_explicit((volatile Shared atomic_int*)location, value, memory_order_relaxed)
#define atomicCmpXchgShared(location, existingValue, desiredValue) \
  atomic_compare_exchange_weak_explicit((volatile Shared atomic_uint*)location, ((Thread uint*)&existingValue), asUint(desiredValue), memory_order_relaxed, memory_order_relaxed)

#define KERNEL_GLOBAL_ARGUMENTS \
  , uint3 thread_position_in_grid [[ thread_position_in_grid ]]
#define KERNEL_THREAD_ARGUMENTS \
  , ushort thread_index_in_threadgroup [[ thread_index_in_threadgroup ]] \
  , ushort3 thread_position_in_threadgroup [[ thread_position_in_threadgroup ]] \
  , ushort3 threads_per_threadgroup [[ threads_per_threadgroup ]]
#define KERNEL_THREADGROUP_ARGUMENTS \
  , ushort3 threadgroup_position_in_grid [[ threadgroup_position_in_grid ]] \
  , ushort3 threadgroups_per_grid [[ threadgroups_per_grid ]]

#define ALIGN(n)            __attribute__((packed)) alignas(n)

struct commonInt8
{
  int4 a, b;
  commonInt8(int x):a(x), b(x)
  {}
};

struct commonUint8
{
  uint4 a, b;
  commonUint8(uint x):a(x), b(x)
  {}
};

struct commonFloat8
{
  float4 a, b;
  commonFloat8(float x):a(x), b()
  {}
};

struct commonInt16
{
  commonInt8 a, b;
  commonInt16(int x):a(x), b(x)
  {}
};

struct commonUint16
{
  commonUint8 a, b;
  commonUint16(int x):a(x), b(x)
  {}
};

struct commonFloat16
{
  commonFloat8 a, b;
  commonFloat16(float x):a(x), b(x)
  {}
};

#endif

#define COMPUTE_SHADER_SCOPE
#define COMPUTE_EPSILON 0.0001f
#define COMPUTE_EPSILON_SQ 0.00000001f
#define DEFAULT_ALIGN   ALIGN(16)

#define NUM_BANKS       8
#define LOG_NUM_BANKS   3

#define paddedIndex(x) x
//#define paddedIndex(x) (x + (x >> NUM_BANKS) + (x >> (2*LOG_NUM_BANKS)))
//#define paddedIndex(x) (x + (x >> (LOG_NUM_BANKS)))
/*inline const uint paddedIndex(const uint n)
{
  //return n + (((n >> NUM_BANKS) + n) >> (LOG_NUM_BANKS << 1));
  return n + (n >> LOG_NUM_BANKS) + (n >> (1+LOG_NUM_BANKS));
}*/

static float sqr(const float x)
{
  return x * x;
}

#pragma pack(push, 4)

typedef struct ALIGN(4)
{
  float val[9];
} Matrix3x3;

#pragma pack(pop)

#define addMatrix3x3(a, b)    for (uint i = 0; i < 9; i++) { (a)->val[i] += (b)->val[i]; }
#define divMatrix3x3(a, b)    for (uint i = 0; i < 9; i++) { (a)->val[i] /= (*b); }
#define copyMatrix3x3(a, b)   for (uint i = 0; i < 9; i++) { (a)->val[i] = (b)->val[i]; }
#define clearMatrix3x3(a, b)  for (uint i = 0; i < 9; i++) { (a)->val[i] = b; }
#define reduceMatrix3x3(o, in)for (uint i = 0; i < 9; i++) { (o)->val[i] = simdReduce((in)->val[i]); }

// this is just a safety measure to make sure the kernel ends and does not end up in an infinite loop
#define INIT_POLL()     short poll_count = 0;
#define POLL_TIMEOUT()  (poll_count++ >= 20000)
#define RESET_POLL()    poll_count = 0

#ifndef ASSUME_FLEXIBLE_VECTOR_ALIGNMENT

#define writeFloat4ToDeviceFloat(vec, scaAddr) \
  (scaAddr)[0] = (vec).x; \
  (scaAddr)[1] = (vec).y; \
  (scaAddr)[2] = (vec).z; \
  (scaAddr)[3] = (vec).w;

#define writeToDevice4x(dst, src) \
  (dst)[0] = (src)[0]; \
  (dst)[1] = (src)[1]; \
  (dst)[2] = (src)[2]; \
  (dst)[3] = (src)[3];

#define readFromDevice4x(dst, src) \
  (dst)[0] = (src)[0]; \
  (dst)[1] = (src)[1]; \
  (dst)[2] = (src)[2]; \
  (dst)[3] = (src)[3];

#define readFromDevice2x(dst, src) \
  (dst)[0] = (src)[0]; \
  (dst)[1] = (src)[1];

#else

#define writeFloat4ToDeviceFloat(vec, scaAddr) \
  *((Device float4*)(scaAddr)) = (vec);

#define writeToDevice4x(dst, src) \
  *((Device float4*)(dst)) = *((Thread float4*)(src));

#define readFromDevice4x(dst, src) \
  *((Thread float4*)(dst)) = *((Device float4*)(src));

#define readFromDevice2x(dst, src) \
  *((Thread float2*)(dst)) = *((Device float2*)(src));

#endif

#endif
