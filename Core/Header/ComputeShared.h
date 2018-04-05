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

#endif