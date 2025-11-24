/*
 * RaptorPhysics
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#ifndef CONSTRAIN_STRUCT_H
#define CONSTRAIN_STRUCT_H

#ifndef COMPUTE_SHADER_SCOPE
#include <Core.h>
#endif

#define CONSTRAIN_OFFSET_BITS 24
#define CONSTRAIN_OFFSET_MASK 0xFFFFFF

#pragma pack(push, 4)

/*
@struct Class representing a constrain offset in an array and number of constrains.
*/
struct ConstrainStruct_t
{
  uint value;
};

typedef struct ConstrainStruct_t ConstrainStruct;

#ifdef COMPUTE_SHADER_SCOPE

/*!@function Extract constrain offset data.*/
static uint constrainOffset(const ConstrainStruct ref)
{
  return ref.value & CONSTRAIN_OFFSET_MASK;
}

/*!@function Extract constrain count data.*/
static uint constrainCount(const ConstrainStruct ref)
{
  return ref.value >> CONSTRAIN_OFFSET_BITS;
}

#endif

#pragma pack(pop)

#endif
