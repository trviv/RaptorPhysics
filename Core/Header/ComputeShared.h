#ifndef COMPUTE_SHARED_H
#define COMPUTE_SHARED_H

#ifndef COMPUTE_SHADER_SCOPE
#include "Root.h"
#endif

/*
@struct Data describing a sub-part of an array.
*/
struct PartitionInfo_t
{
  /*@member Offset.*/
  uint  offset;
  /*@member Count.*/
  uint  count;
};

typedef struct PartitionInfo_t PartitionInfo;

#endif