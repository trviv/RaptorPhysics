#ifndef LINEAR_SOLVER_SHADER
#define LINEAR_SOLVER_SHADER

Kernel void linearSolver(
  Device VariableType*    newValues,
  Const ConstrainStruct*  constrainNodes,
  Const IndexType*        indexArray,
  Const CoefficientType*  coefficients,
  Const VariableType*     oldValues,
  Const VariableType*     constantValues,
  const uint nodeCount)
{
  const uint index = threadIndex();
  const VariableType weight = (VariableType)(3. / 4.);

  if (index < nodeCount)
  {
    VariableType sum = 0;
    uint count = constrainCount(constrainNodes[index]);
    const uint offset = constrainOffset(constrainNodes[index]);
    VariableType selfCoef = coefficients[offset];
    for (int i = 1; i < count; i++)
    {
      sum += coefficients[offset + i] * oldValues[indexArray[offset + i]];
    }
    const VariableType one = 1;
    newValues[index] = (one - weight)*oldValues[index] + weight*(constantValues[index] - sum) / selfCoef;
  }
}

#endif