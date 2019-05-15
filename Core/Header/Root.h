#ifndef ROOT_H
#define ROOT_H

// for apple platform
#if __APPLE__
#define ENV_WIN         0   // for windows enviornment
#define ENV_APPLE       1   // for windows enviornment

#else
#define ENV_WIN         1   // for windows enviornment
#define ENV_APPLE       0   // for windows enviornment

#endif

#define REN_GL          1 //open gl renderer
#define USING_APPROX    0 //for using approximation methods
#define PREC_DOUBLE     0 //for using double as primary data type

// common include files
#include <map>
#include <cmath>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <fstream>
#include <stdint.h>
#include <assert.h>
#include <string.h>
#include <iostream>

#if ENV_WIN
#include <omp.h>
#include <dvec.h>
#include <time.h>
#include <conio.h>
#include <windows.h>

#elif ENV_APPLE
#include <_types.h>
#include <stdlib.h>
#include <termios.h>
#include <pthread.h>
#include <sys/time.h>

#endif

#if REN_GL
#include <glew.h>

#if ENV_WIN
#include <freeglut.h>

#endif

#endif

#if PREC_DOUBLE
#define PREC_FLOAT  0
#define EPSILON     DBL_EPSILON
#define MIN_R       -DBL_MAX
#define MAX_R       DBL_MAX
#define INF         1e10
#define MIN         .000001

typedef double real;

#else
#define PREC_FLOAT    1
#define EPSILON       FLT_EPSILON
#define MIN_R         -FLT_MAX
#define MAX_R         FLT_MAX
#define INF           1e30f
#define MIN           1e-10f

typedef float real;
#endif

#if ENV_APPLE
#undef M_PI
#undef M_2_PI
#undef M_PI_2

#else
typedef uint8_t   uchar;
typedef uint16_t  ushort;
typedef uint32_t  uint;
typedef uint64_t  ulong;
typedef uint16_t  half;

#endif

#define M_PI_F    real(3.1415926535897932384626433832795)
#define M_PI      real(3.1415926535897932384626433832795)
#define M_PI_180  real(0.017453292519943295769236907684883)
#define M_2_PI    real(6.283185307179586476925286766560)
#define M_2_PI_F  real(6.283185307179586476925286766560)
#define M_PI_2    real(.5)*M_PI
#define M_PI_2_F  real(.5)*M_PI_F
#define INV_PI    real(0.31830988618379067154)
#define INV_TWOPI real(0.15915494309189533577)

#if ENV_WIN
#define FORCE_INLINE  __forceinline

#elif ENV_APPLE
#define FORCE_INLINE  __inline

#endif

#define prompt(X) assert(X)

#if defined(__CUDACC__) // NVCC
#define ALIGN(n)  __align__(n)
#elif defined(__GNUC__) || defined(OPENCL) // GCC or OpenCL
#define ALIGN(n)  __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
#define ALIGN(n)  __declspec(align(n))
#else
#error "Please provide a definition for MY_ALIGN macro for your host compiler!"
#endif

#define DEFAULT_ALIGN ALIGN(16)

static std::ostream &cout = std::cout;

#endif
