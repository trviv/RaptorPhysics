#ifndef __DISTANCE_CONSTRAIN
#define __DISTANCE_CONSTRAIN

#include "Constrain.h"

ALIGN(16) class DistanceConstrain :
public ConstrainSolver < __int32, real, Real3 >
{
protected:
  typedef Counter IndexType;
  typedef real    CoefType;
  typedef Real3   ValueType;

  typedef ConstrainSolver < IndexType, CoefType, ValueType > Constrain;

public:

  CU_DEV void getDelta(ValueType& del, const IndexType index,
    const IndexType connection_index, const IndexType offset,
    const ConstrainBuffer buffer_index);
  void solve();
};

#endif