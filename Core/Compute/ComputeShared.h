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
    float3  position;
    struct
    {
      uint  positionUint[3];
      uint  dataId;
    };
  };
};

typedef struct PositionStruct_t PositionStruct;

#pragma pack(pop)

#endif
