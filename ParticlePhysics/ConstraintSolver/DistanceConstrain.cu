#include "DistanceConstrain.h"

CU_DEV void DistanceConstrain::getDelta(ValueType& del, const IndexType index,
  const IndexType connection_index, const IndexType offset,
  const ConstrainBuffer buffer_index)
{
  del = 0;
  const real w1 = getMass()[index];
  const real w12 = w1 + getMass()[connection_index];
  if (w12 > real(0.0000001))
  {
    del = getValue(connection_index, buffer_index) -
      getValue(index, buffer_index);
    del *= (getDistance()[offset] / del.length() - real(1));
    del *= -w1 / w12;
  }
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
  Real3 del;
  Counter count = constrain->getConstrain(index).count();
  Counter offset = constrain->getConstrain(index).offset();
  const real constrain_count = (count <= 1) ? 1 : count;

  for (count = count - 1; count >= 0; count--)
  {
    constrain->getDelta(del, index, constrain->getIndex(offset), offset,
      old_value);
    sum += del;
    offset++;
  }
  constrain->getValue(index, new_value) =
    constrain->getValue(index, old_value) + sum*(
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
  //scale(getDelPosition(), getDelPosition(), real(1) / del_t, getNodeCount());
  //DeviceEntity<Real3>::copy(getPosition(), getValueBuffer(DEF), getNodeCount());
  differentiate();
}