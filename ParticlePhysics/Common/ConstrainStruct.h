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
  uint value;
};

typedef struct ConstrainStruct_t ConstrainStruct;

#ifdef COMPUTE_SHADER_SCOPE

/*@function Extract constrain offset data.*/
static uint constrainOffset(const ConstrainStruct ref)
{
  return ref.value & CONSTRAIN_OFFSET_MASK;
}

/*@function Extract constrain count data.*/
static uint constrainCount(const ConstrainStruct ref)
{
  return ref.value >> CONSTRAIN_OFFSET_BITS;
}

#endif

#endif
