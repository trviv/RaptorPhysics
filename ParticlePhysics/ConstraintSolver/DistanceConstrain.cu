#include "DistanceConstrain.h"

CU_DEV void DistanceConstrain::getDelta(ValueType& del, const IndexType index,
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
  del *= (real(1) - getDistance()[offset] / del.length());
  del *= -w1 / w12;
  //del = (-w1*(real(1) - getDistance()[offset] / del.length()) / (w1 + w2));
}

void DistanceConstrain::add(const IndexType index, const IndexType connection,
  const CoefType coef, const real distance,
  const real inv_mass)
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

void DistanceConstrain::exportToDevice(__int8** device_additional_memory,
  Counter additional_size, Counter baseSize)
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
    4 * sizeof(Real3)*constraints.size();

  Constrain::exportToDevice(&device_memory, total_size, baseSize);

  device_mass = (real*)device_memory;
  device_distance = (device_mass + point_mass.size());
  device_position = (Real3*)(device_distance + flat_distance.size());
  device_velocity = (Real3*)(device_position + getNodeCount());
  device_force = (Real3*)(device_velocity + getNodeCount());
  device_del_pos = (Real3*)(device_force + getNodeCount());
  if (additional_size)
  {
    *device_additional_memory = ((__int8*)device_del_pos +
      getNodeCount() * sizeof(Real3));
  }

  std::cout << "Rigid Alloc:\n";
  std::cout << "Base: " << baseSize <<
    "\tAddr: " << (__int64)constrain_alloc << "\n";
  std::cout << "Mass: " << sizeof(real)*point_mass.size() <<
    "\tAddr: " << (__int64)device_mass << "\n";
  std::cout << "Distance: " << sizeof(real)*flat_distance.size() <<
    "\tAddr: " << (__int64)device_distance <<
    "\tDiff: " << (__int64)device_distance - (__int64)device_mass << "\n";
  std::cout << "Position: " << sizeof(Real3)*getNodeCount() <<
    "\tAddr: " << (__int64)device_position <<
    "\tDiff: " << (__int64)device_position - (__int64)device_distance << "\n";
  std::cout << "Velocity: " << sizeof(Real3)*getNodeCount() <<
    "\tAddr: " << (__int64)device_velocity <<
    "\tDiff: " << (__int64)device_velocity - (__int64)device_position << "\n";
  std::cout << "Force: " << sizeof(Real3)*getNodeCount() <<
    "\tAddr: " << (__int64)device_force <<
    "\tDiff: " << (__int64)device_force - (__int64)device_velocity << "\n";
  std::cout << "Del pos: " << sizeof(Real3)*getNodeCount() <<
    "\tAddr: " << (__int64)device_del_pos <<
    "\tDiff: " << (__int64)device_del_pos - (__int64)device_force << "\n";
  if (additional_size > 0)
  {
    std::cout << "Additional: " << additional_size <<
      "\tAddr: " << (__int64)device_additional_memory <<
      "\tDiff: " << (__int64)device_additional_memory -
      (__int64)device_del_pos << "\n";
  }

  DeviceEntity<real>::exportToDevice(&point_mass[0], device_mass,
    point_mass.size());
  DeviceEntity<real>::exportToDevice(&flat_distance[0], device_distance,
    flat_distance.size());
  DeviceEntity<Real3>::copy(device_position, device_value_arrays,
    getNodeCount());
  DeviceEntity<Real3>::set(device_velocity, 0, getNodeCount());
  DeviceEntity<Real3>::set(device_force, 0, getNodeCount());
  DeviceEntity<Real3>::set(device_del_pos, 0, getNodeCount());
  DeviceEntity<DistanceConstrain>::exportToDevice(this,
    (DistanceConstrain*)constrain_alloc);
}

CU_KER void distanceSolver(
  DistanceConstrain* constrain,
  const Counter iteration)
{
  Counter index = threadIndex;
  if (index >= constrain->getNodeCount()) return;

  const ConstrainBuffer old_value = ((iteration & 1) == 0) ? DEF : VAR0;
  const ConstrainBuffer new_value = ((old_value == DEF) ? VAR0 : DEF);
  Real3 sum(0);
  Counter count = constrain->getConstrain(index).count();
  Counter offset = constrain->getConstrain(index).offset();
  Real3 del;
  offset++;
  real constrain_count = (count <= 1) ? 1 : count - 1;
  for (count = count - 2; count >= 0; count--)
  {
    constrain->getDelta(del, index, constrain->getIndex(offset), offset,
      old_value);
    sum += del;
    offset++;
  }
  constrain->getValue(index, new_value) =
    constrain->getValue(index, old_value) + sum*(//constrain->del_t*
    constrain->successiveOverRealaxation / constrain_count);
}

void DistanceConstrain::solve()
{
  integrate();
  dim3 threads;
  dim3 blocks;
  configureGrid(blocks, threads);
  DeviceEntity<Real3>::copy(getValueBuffer(DEF), getPosition(), getNodeCount());
  for (Counter i = 0; i < iterations; i++)
  {
    distanceSolver << <blocks, threads >> >((DistanceConstrain*)constrain_alloc, i);
    cudaDeviceSynchronize();
    CU_PROMPT;
  }

  if ((getIterations() & 1) == 0)
  {
    DeviceEntity<Real3>::copy((Real3*)getValueBuffer(DEF),
      (Real3*)getValueBuffer(VAR0), getNodeCount());
  }
  sum(getDelPosition(), getValueBuffer(DEF), getPosition(), getNodeCount(), true);
  //scale(getDelPosition(), getDelPosition(), del_t, getNodeCount());
  DeviceEntity<Real3>::copy(getPosition(), getValueBuffer(DEF), getNodeCount());
  differentiate();
}