#ifndef COMPUTE_SHARED_H
#define COMPUTE_SHARED_H

#ifndef COMPUTE_SHADER_SCOPE
#include <Vector/Real3.h>
#endif

#pragma pack(push, 4)

typedef float3 Color3;

/*!
@struct Data describing an array sub-part.
*/
struct ALIGN(8) PartitionInfo_t
{
  /*!@member Offset.*/
  uint  offset;

  /*!@member Count.*/
  uint  count;

#ifndef COMPUTE_SHADER_SCOPE

  PartitionInfo_t()
  {}

  PartitionInfo_t(uint init)
  {
    offset = init;
    count = init;
  }

  uint end()const
  {
    return offset + count;
  }
#endif

};

typedef struct PartitionInfo_t PartitionInfo;

struct ALIGN(8) SortNode32_t
{
  uint key;
  uint value;
};

typedef struct SortNode32_t SortNode32;

#if defined(COMPUTE_SHADER_SCOPE) && !defined(ComputeUtilSkipParallelPrimitives)

static SortNode32 defaultSortNode()
{
  SortNode32 ret;
  ret.key = -1;
  ret.value = -1;
  return ret;
}

#endif


/*!
@struct Axis aligned bounding box data.
*/
struct DEFAULT_ALIGN XAB_t
{
  union
  {
    float3  min;
    float   reserved1[4];
  };
  union
  {
    float3  max;
    float   reserved2[4];
  };
};

typedef struct XAB_t XAB;

#define mergeXAB(a, b)  { (a)->min = min((a)->min, (b)->min); (a)->max = max((a)->max, (b)->max);}
#define divXAB(a, b)    { (a)->min /= (*b); (a)->max /= (*b);}
#define copyXAB(a, b)   { (a)->min = (b)->min; (a)->max = (b)->max;}
#define clearXAB(a, b)  { (a)->min = INFINITY; (a)->max = -INFINITY;}
#define reduceXAB(o, i) { o.min = simdMin(i.min); o.max = simdMax(i.max);}

#define mergeFloat(a, b)  { *a = max(*a, *b);}
#define reduceFloat(o, i) { o = simdMax(i);}


/*!
@struct Generic template structure to store position, and and some associated uint data.
        Should serve as the basis for similar datatypes for individual projects.
*/
struct DEFAULT_ALIGN PositionStruct_t
{
  union
  {
    struct
    {
      float3  position;
    };
    struct
    {
      uint  positionUint[3];
      uint  dataId;
    };
  };
};

typedef struct PositionStruct_t PositionStruct;


#ifdef COMPUTE_SHADER_SCOPE

inline uint encode32Bits(uint x)
{
  //........ ........ ......12 3456789A  //x
  //....1..2 ..3..4.. 5..6..7. .8..9..A  //x after interleaving bits

  //......12 3456789A ......12 3456789A  //x ^ (x << 16)
  //11111111 ........ ........ 11111111  //0x FF 00 00 FF
  //......12 ........ ........ 3456789A  //x = (x ^ (x << 16)) & 0xFF0000FF;

  //......12 ........ 3456789A 3456789A  //x ^ (x <<  8)
  //......11 ........ 1111.... ....1111  //0x 03 00 F0 0F
  //......12 ........ 3456.... ....789A  //x = (x ^ (x <<  8)) & 0x0300F00F;

  //..12..12 ....3456 3456.... 789A789A  //x ^ (x <<  4)
  //......11 ....11.. ..11.... 11....11  //0x 03 0C 30 C3
  //......12 ....34.. ..56.... 78....9A  //x = (x ^ (x <<  4)) & 0x030C30C3;

  //....1212 ..3434.. 5656..78 78..9A9A  //x ^ (x <<  2)
  //....1..1 ..1..1.. 1..1..1. .1..1..1  //0x 09 24 92 49
  //....1..2 ..3..4.. 5..6..7. .8..9..A  //x = (x ^ (x <<  2)) & 0x09249249;

  //........ ........ ......11 11111111  //0x000003FF

  x = (x ^ (x << 16)) & 0xFF0000FF;
  x = (x ^ (x << 8)) & 0x0300F00F;
  x = (x ^ (x << 4)) & 0x030C30C3;
  x = (x ^ (x << 2)) & 0x09249249;

  return x;
}

#define MORTON_CODE_MASK 1023

inline uint encode32BitMortonCode(const int3 quantizedPosition)
{
  const uint x = quantizedPosition.x & MORTON_CODE_MASK;
  const uint y = quantizedPosition.y & MORTON_CODE_MASK;
  const uint z = quantizedPosition.z & MORTON_CODE_MASK;

  return encode32Bits(x) | (encode32Bits(y) << 1) | (encode32Bits(z) << 2);
}

inline uint decode32Bits(uint x)
{
  x &= 0x09249249;                  // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
  x = (x ^ (x >>  2)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
  x = (x ^ (x >>  4)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
  x = (x ^ (x >>  8)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
  x = (x ^ (x >> 16)) & 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210

  return x;
}

inline int3 decode32BitMortonCode(const uint mortonCode)
{
  return constructInt3(decode32Bits(mortonCode), decode32Bits(mortonCode >> 1), decode32Bits(mortonCode >> 2));
}

#endif


/*!
@struct Bounding volume hierarchy leaf data.
*/
struct ALIGN(8) BVHLeafInfo_t
{
  uint mortonCode;
  uint index;
};

typedef struct BVHLeafInfo_t BVHLeafInfo;


/*!
@struct Bounding volume hierarchy internal node data.
*/
struct ALIGN(8) BVHNodeInfo_t
{
  uint child[2];
};

typedef struct BVHNodeInfo_t BVHNodeInfo;

#pragma pack(pop)

#endif
