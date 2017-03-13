#ifndef __RIGID_CONSTRAIN
#define __RIGID_CONSTRAIN

#include "DistanceConstrain.h"

class RigidConstrain;

CU_KER void rigidSolver(
  RigidConstrain* constrain,
  const Counter step,
  const Counter length = 0);

static CU_KER void adjointMatrix(real* value, const Counter length)
{
  Counter index = threadIndex;
  real minor;
  Counter i = index / 3, j = index % 3;
  if (index < length)
  {
    minor = value[((i + 1) % 3) * 3 + ((j + 1) % 3)] *
      value[((i + 2) % 3) * 3 + ((j + 2) % 3)] -
      value[((i + 1) % 3) * 3 + ((j + 2) % 3)] *
      value[((i + 2) % 3) * 3 + ((j + 1) % 3)];
    if (i == 1)
    {
      minor *= -1;
    }
    minor *= mPow<real>(-1, i + j);
  }
  __syncthreads();
  if (index < length)
  {
    value[i + j * 3] = minor;
  }
}

static CU_KER void matrix(real* value)
{
  for (int i = 0; i < 3; i++)
    printf("%f %f %f\n", value[i * 3 + 0], value[i * 3 + 1], value[i * 3 + 2]);
}

ALIGN(16) class RigidConstrain : public DistanceConstrain
{
public:

  CU_DEV void getDelta(ValueType& del, const IndexType index,
    const IndexType connection_index, const IndexType offset,
    const ConstrainBuffer buffer_index)
  {
    del = 0;
    real w1 = getMass()[index];
    real w2 = getMass()[connection_index];
    if (w1 + w2 < 0.0000001) return;

    del = getValue(index, buffer_index) -
      getValue(connection_index, buffer_index);
    del *= -w1*(1.f - getDistance()[offset] / del.length()) / (w1 + w2);
  }

  FORCE_INLINE void add(const IndexType index, const IndexType connection,
    const CoefType coef, const real distance = 0,
    const real inv_mass = 0)
  {
    Constrain::add(index, connection, coef);
    if (index == connection)
    {
      expand<real>(index + offset, point_mass);
      point_mass[index + offset] = inv_mass;
    }
    expand<std::vector<real>>(index + offset, point_distance);
    point_distance[index + offset].push_back(distance);
  }

  void exportToDevice(__int8** device_additional_memory = NULL,
    int additional_size = 0, int baseSize = 0);

  void solve();
};

#endif