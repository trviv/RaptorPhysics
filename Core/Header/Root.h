#ifndef RX_ROOT
#define RX_ROOT

#define RX_ENV_WIN      1 //for windows enviornment

#define RX_PLF_PC       1 //for pc platform

#define RX_REN_GL       1 //open gl renderer
#define RX_REN_D3       0 //directx

#define RX_CUDA         1
#define RX_USING_APPROX 0 //for using approximation methods

#define RX_CUSTOM_MM

#define RX_PREC_DOUBLE  0 //for using double as primary data type

/*
#define RX_NS_CORE
#define RX_NS_RAY RX_NS_CORE
#define RX_NS_PHOTON RX_NS_CORE
*/

#if RX_PLF_PC
//#include <map>
#include <omp.h>
//#include <dvec.h>
//#include <math.h>
#include <conio.h>
#include <float.h>
#include <fstream>
#include <assert.h>
#include <iostream>
#endif

#if RX_ENV_WIN
#include <windows.h>
//#include <time.h>
#endif

#if RX_REN_GL
#include <glew.h>
#include <freeglut.h>
#endif

#if RX_REN_D3
#include <d3d9.h>
#endif

#ifdef _X86_
#define RX_32
#else _AMD64_
#define RX_64
#endif

#if RX_CUDA

//#include <cuda.h>
#include <cuda_runtime.h>

//use only single precision if cuda enabled
#if RX_PREC_DOUBLE
#undef  RX_PREC_DOUBLE
#define RX_PREC_DOUBLE  0
#endif

#define CU_FI   __forceinline__
#define CU_NI   __noinline__
#define CU_DEV  __device__
#define CU_HOST __host__
#define CU_DEV_HOST __device__ __host__
#define CU_KER  __global__
#define CU_SHA  __shared__
#define CU_PROMPT checkCudaErrors(cudaGetLastError());

#endif

#define NAMELEN 16
#define CDEF

#if RX_PREC_DOUBLE
#define RX_PREC_FLOAT 0
#define EPSILON DBL_EPSILON
#define MIN_R -DBL_MAX
#define MAX_R DBL_MAX
#define INF 1e10
#define MIN .000001

typedef double real;
#else
#define RX_PREC_FLOAT 1
#define EPSILON FLT_EPSILON
#define MIN_R -FLT_MAX
#define MAX_R FLT_MAX
#define INF 1e10f
#define MIN .0001f

typedef float real;
#endif

typedef unsigned long ULong;
typedef unsigned short UShort;
typedef long Long;
typedef __int32 Counter;
typedef __int16 Index;
typedef float Float;
typedef unsigned __int8 Byte;
typedef unsigned __int32 UInt32;

#define INV_RAND_MAX real(1.0/32768.0)
#define INV_RAND_MAX_F Float(1.0f/32768.0f)
#define M_PI_F Float(3.1415926535897932384626433832795)
#define M_PI real(3.1415926535897932384626433832795)
#define M_PI_180 real(0.017453292519943295769236907684883)
#define M_2_PI real(6.283185307179586476925286766560)
#define M_PI_2 real(.5)*M_PI
#define M_PI_2_F Float(.5)*M_PI_F
#define M_2_PI_F Float(6.283185307179586476925286766560)
#define INV_PI  real(0.31830988618379067154)
#define INV_TWOPI  real(0.15915494309189533577)

#if RX_PLF_PC
#define FORCE_INLINE __forceinline
#define prompt(X) assert(X)
#endif

#if defined(__CUDACC__) // NVCC
#define ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
#define ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
#define ALIGN(n) __declspec(align(n))
#else
#error "Please provide a definition for MY_ALIGN macro for your host compiler!"
#endif

static std::ostream &cout = std::cout;

#endif