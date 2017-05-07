#ifndef __RIGID_CONSTRAIN
#define __RIGID_CONSTRAIN

#include "DistanceConstrain.h"

class RigidConstrain;

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
/*
static CU_KER void matrix(real* value)
{
Counter index = threadIndex;
//for (Counter i = 0; i < 3; i++)
//debugLog("%f %f %f\n", value[i * 3 + 0], value[i * 3 + 1], value[i * 3 + 2]);
debugLog("%f ", value[index]);
}
*/

static void showMatrix(const Counter w, const Counter h, const real* device_mat)
{
  real* host_mat = new real[w*h];
  DeviceEntity<real>::importToHost(host_mat, device_mat, w*h);
  for (Counter i = 0; i < h; i++)
  {
    printf("Row:%d\n", i);
    for (Counter j = 0; j < w; j++)
    {
      printf("%f ", host_mat[i*w + j]);
    }
    printf("\n");
  }
  delete host_mat;
}

ALIGN(16) class RigidConstrain : public DistanceConstrain
{
public:

  CU_DEV void getDelta(ValueType& del, const IndexType index,
    const IndexType connection_index, const IndexType offset,
    const ConstrainBuffer buffer_index);
  void solve();
};

#endif