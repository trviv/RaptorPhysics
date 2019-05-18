#ifndef COMPUTE_SHARED_H
#define COMPUTE_SHARED_H

#ifndef COMPUTE_SHADER_SCOPE
#include "Root.h"
#endif

/*
@struct Data describing an array sub-part.
*/
struct PartitionInfo_t
{
  /*@member Offset.*/
  uint  offset;

  /*@member Count.*/
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

struct SortNode32_t
{
  uint key;
  uint value;
};

typedef struct SortNode32_t SortNode32;

#ifdef COMPUTE_SHADER_SCOPE

SortNode32 defaultSortNode()
{
  SortNode32 ret;
  ret.key = -1;
  ret.value = -1;
  return ret;
}

#endif

#endif
