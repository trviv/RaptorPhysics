/*
 * Copyright (c) Vivek Trivedi, 2025
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#ifndef CONSTRAIN_H
#define CONSTRAIN_H

#include <Core.h>
#include "../Common/ConstrainStruct.h"

/*
@class Class representing a constrain offset in an array and number of constrains.
*/
class Constrain : protected ConstrainStruct
{
public:
  Constrain(const uint offset = 0, const uint count = 0)
  {
    value = 0;
    setOffset(offset);
    setCount(count);
  }

  uint count()
  {
    return value >> CONSTRAIN_OFFSET_BITS;
  }

  uint offset()
  {
    return value & CONSTRAIN_OFFSET_MASK;
  }

  void setOffset(const uint offset)
  {
    value = ((value & (-1 ^ CONSTRAIN_OFFSET_MASK)) | (offset & CONSTRAIN_OFFSET_MASK));
  }

  void setCount(const uint count)
  {
    value = ((value & CONSTRAIN_OFFSET_MASK) | (count << CONSTRAIN_OFFSET_BITS));
  }
};

#endif