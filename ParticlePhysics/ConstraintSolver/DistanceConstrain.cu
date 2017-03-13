#include "DistanceConstrain.h"

CU_KER void distanceSolver(
  DistanceConstrain* constrain,
  const Counter iteration)
{
  const ConstrainBuffer old_value = ((iteration & 1) == 0) ? DEF : VAR0;
  const ConstrainBuffer new_value = ((old_value == DEF) ? VAR0 : DEF);
  Real3 sum;

  Counter index = threadIndex;
  if (index < constrain->getNodeCount())
  {
    sum = 0;
    Counter count = constrain->getConstrain(index).count();
    Counter offset = constrain->getConstrain(index).offset();
    Real3 del;
    offset++;
    float constrain_count = count;
    for (count = count - 2; count >= 0; count--)
    {
      constrain->getDelta(del, index, constrain->getIndex(offset), offset,
        old_value);
      sum += del;
      offset++;
    }
    constrain->getValue(index, new_value) =
      constrain->getValue(index, old_value) + sum*(constrain->del_t*
      constrain->successiveOverRealaxation / (constrain_count - 1));
  }
}