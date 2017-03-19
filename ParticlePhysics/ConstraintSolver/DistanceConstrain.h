#ifndef __DISTANCE_CONSTRAIN
#define __DISTANCE_CONSTRAIN

#include "Constrain.h"

class DistanceConstrain;

CU_KER void distanceSolver(
  DistanceConstrain* constrain,
  const Counter total_iterations);

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
    const ConstrainBuffer buffer_index)
  {
    del = 0;
    const real w1 = getMass()[index];
    const real w2 = getMass()[connection_index];
    const real w12 = w1 + w2;
    if (w12 < real(0.0000001)) return;

    del = getValue(index, buffer_index) -
      getValue(connection_index, buffer_index);
    del *= real(1) - getDistance()[offset] / del.length();
    del *= -w1 / w12;
    //del = (-w1*(real(1) - getDistance()[offset] / del.length()) / (w1 + w2));
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
    Counter additional_size = 0, Counter baseSize = 0)
  {
    __int8* device_memory = NULL;
    if (!baseSize) baseSize = sizeof(DistanceConstrain);

    std::vector<real> flat_distance;
    for (Counter i = 0; i < point_distance.size(); i++)
      for (Counter j = 0; j < point_distance[i].size(); j++)
        flat_distance.push_back(point_distance[i][j]);

    IndexType total_size = additional_size +
      sizeof(real)*point_mass.size() +
      sizeof(real)*flat_distance.size() +
      3 * sizeof(Real3)*constraints.size();

    Constrain::exportToDevice(&device_memory, total_size, baseSize);

    device_mass = (real*)device_memory;
    device_distance = (device_mass + point_mass.size());
    device_position = (Real3*)(device_distance + flat_distance.size());
    device_velocity = (Real3*)(device_position + getNodeCount());
    device_force = (Real3*)(device_velocity + getNodeCount());
    if (additional_size)
    {
      *device_additional_memory = ((__int8*)device_force +
        getNodeCount() * sizeof(Real3));
    }

    DeviceEntity<real>::exportToDevice(&point_mass[0], device_mass,
      point_mass.size());
    DeviceEntity<real>::exportToDevice(&flat_distance[0], device_distance,
      flat_distance.size());
    DeviceEntity<Real3>::copy(device_position, device_value_arrays,
      getNodeCount());
    DeviceEntity<Real3>::set(device_velocity, 0, getNodeCount());
    DeviceEntity<Real3>::set(device_force, 0, getNodeCount());
    DeviceEntity<DistanceConstrain>::exportToDevice(this,
      (DistanceConstrain*)constrain_alloc);
  }

  void solve()
  {
    integrate();
    dim3 threads;
    dim3 blocks;
    configureGrid(blocks, threads);
    for (int i = 0; i < iterations; i++)
    {
      distanceSolver << <blocks, threads >> >((DistanceConstrain*)constrain_alloc, i);
      cudaDeviceSynchronize();
      CU_PROMPT;
    }
    if ((getIterations() & 1) == 0)
    {
      DeviceEntity<Real3>::copy((Real3*)getValueBuffer(DEF),
        (Real3*)getValueBuffer(VAR0), getNodeCount());
      cudaDeviceSynchronize();
    }
    differentiate();
  }
};

#endif