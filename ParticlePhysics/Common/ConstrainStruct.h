#ifndef CONSTRAIN_STRUCT_H
#define CONSTRAIN_STRUCT_H

#ifndef COMPUTE_SHADER_SCOPE
#include <Core.h>
#endif

#define CONSTRAIN_OFFSET_BITS 24
#define CONSTRAIN_OFFSET_MASK 0xFFFFFF

/*
@struct Class representing a constrain offset in an array and number of constrains.
*/
struct ConstrainStruct_t
{
  unsigned int value;
};

typedef struct ConstrainStruct_t ConstrainStruct;

static unsigned int constrainOffset(const ConstrainStruct ref)
{
  return ref.value & CONSTRAIN_OFFSET_MASK;
}

static unsigned int constrainCount(const ConstrainStruct ref)
{
  return ref.value >> CONSTRAIN_OFFSET_BITS;
}

#endif