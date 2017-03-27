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

  void add(const IndexType index, const IndexType connection,
    const CoefType coef, const real distance = 0,
    const real inv_mass = 0);

  void exportToDevice(__int8** device_additional_memory = NULL,
    Counter additional_size = 0, Counter baseSize = 0);

  void solve();
};

#endif