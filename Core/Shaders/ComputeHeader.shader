#ifndef COMPUTE_HEADER_SHADER
#define COMPUTE_HEADER_SHADER

#ifndef USE_METAL_COMPUTE

#define constantKernelInput(type, variableName) const type variableName
#define atomicKernelInput(type, variableName) Device type *variableName

#define threadIndex()       get_global_id(0)
#define threadLocalIndex()  get_local_id(0)
#define threadGroupSize()   get_local_size(0)
#define threadGroupIndex()  get_group_id(0)
#define threadGroupCount()  get_num_groups(0)
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
#define constructFloat2     (float2)
#define constructFloat3     (float3)
#define constructFloat4     (float4)
#define constructUint2      (uint2)
#define constructUint3      (uint3)
#define constructInt2       (int2)
#define constructInt3       (int3)

#define convertShort2(a)    convert_short2(a)
#define convertUshort4(a)   convert_ushort4(a)
#define convertInt3(a)      convert_int3(a)
#define asUchar4(x)         as_uchar4(x)
#define asFloat(x)          as_float(x)
#define simdAll(x)          assert
#define simdFirst(x)        assert
#define simdIsFirst()       assert
#define simdReduce(x)       assert
#define simdScan(x)         assert
#define simdMin(x)          assert
#define simdMax(x)          assert
#define selectInput2(x)     (uint2)x
#define selectInput3(x)     (uint3)(x)

#define atomicLoad(location)          atomic_or  ((Device uint*)location, 0)
#define atomicStore(location, value)  atomic_xchg((Device uint*)location, value)
#define atomicAdd(location, value)    atomic_add (location, value)
#define atomicMax(location, value)    atomic_max (location, value)

#define KERNEL_GLOBAL_ARGUMENTS
#define KERNEL_THREAD_ARGUMENTS
#define KERNEL_THREADGROUP_ARGUMENTS

#define ALIGN(n)            __attribute__((aligned(n))) __attribute__((packed))

#else

#define constantKernelInput(type, variableName) Const type& variableName
#define atomicKernelInput(type, variableName) Device atomic_##type *variableName

#define threadIndex()       thread_position_in_grid[0]
#define threadLocalIndex()  thread_index_in_threadgroup
#define threadGroupSize()   threads_per_threadgroup[0]
#define threadGroupIndex()  threadgroup_position_in_grid[0]
#define threadGroupCount()  threadgroups_per_grid[0]
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
#define constructFloat2     float2
#define constructFloat3     float3
#define constructFloat4     float4
#define constructUint2      uint2
#define constructUint3      uint3
#define constructInt2       int2
#define constructInt3       int3

#define convertShort2(a)    short2(a)
#define convertUshort4(a)   ushort4(a)
#define convertInt3(a)      int3(a)
#define asUchar4(x)         as_type<uchar4>(x)
#define asFloat(x)          as_type<float>(x)
#define simdAll(x)          simd_all(x)
#define simdFirst(x)        simd_broadcast_first(x)
#define simdIsFirst()       simd_is_first()
#define simdReduce(x)       simd_sum(x)
#define simdScan(x)         simd_prefix_inclusive_sum(x)
#define simdMin(x)          simd_min(x)
#define simdMax(x)          simd_max(x)
#define selectInput2(x)     bool2(x)
#define selectInput3(x)     bool3(x)

#define atomicLoad(location)          atomic_fetch_or_explicit((Device atomic_uint*)location, 0, memory_order_relaxed)
#define atomicStore(location, value)  atomic_exchange_explicit((Device atomic_uint*)location, value, memory_order_relaxed)
#define atomicAdd(location, value)    atomic_fetch_add_explicit((Device atomic_uint*)location, value, memory_order_relaxed)
#define atomicMax(location, value)    atomic_fetch_max_explicit((Shared atomic_uint*)location, value, memory_order_relaxed)

#define KERNEL_GLOBAL_ARGUMENTS \
  , uint3 thread_position_in_grid [[ thread_position_in_grid ]]
#define KERNEL_THREAD_ARGUMENTS \
  , ushort thread_index_in_threadgroup [[ thread_index_in_threadgroup ]] \
  , ushort3 threads_per_threadgroup [[ threads_per_threadgroup ]]
#define KERNEL_THREADGROUP_ARGUMENTS \
  , ushort3 threadgroup_position_in_grid [[ threadgroup_position_in_grid ]] \
  , ushort3 threadgroups_per_grid [[ threadgroups_per_grid ]]

#define ALIGN(n)            __attribute__((packed)) alignas(n)

#endif

#define COMPUTE_SHADER_SCOPE
#define COMPUTE_EPSILON 0.0001f
#define DEFAULT_ALIGN   ALIGN(16)

#define NUM_BANKS       16
#define LOG_NUM_BANKS   4

inline const uint paddedIndex(const uint n)
{
  return n;
  //return n + (((n >> NUM_BANKS) + n) >> (LOG_NUM_BANKS << 1));
  //return n + (n >> LOG_NUM_BANKS) + (n >> (1+LOG_NUM_BANKS));
}

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

#endif
