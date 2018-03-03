#ifndef DISTANCE_SOLVER_SHADER
#define DISTANCE_SOLVER_SHADER

#define successiveOverRealaxation 1.5f

VariableType getDelta(
  const Thread VariableType    selfOldValue,
  const Thread VariableType    otherOldValue,
  const Thread CoefficientType coefficient)
{
  VariableType delta = otherOldValue - selfOldValue;
  const float deltaLength = length(delta);
  if (deltaLength > COMPUTE_EPSILON)
  {
    delta *= ((1.f - (coefficient / deltaLength)) * .5f);
  }
  return delta;
}

Kernel void distanceSolverSpring(
  Device ParticleStruct*            newPositions,
  const Device ParticleStruct*      oldPositions,
  const Device ConstrainStruct*     constrainNodes,
  const Device IndexType*           indexArray,
  const Device CoefficientType*     coefficients,
  const uint                        nodeCount)
{
  const uint index = threadIndex();

  if (index < nodeCount)
  {
    const uint offset = constrainOffset(constrainNodes[index]);
    const CoefficientType selfCoef = coefficients[offset];
    const VariableType selfOldValue = oldPositions[index].position;

    newPositions[index].position = selfOldValue;

    if (selfCoef) // if self movememnt allowed
    {
      VariableType sum = 0;
      const uint count = constrainCount(constrainNodes[index]);

      for (uint i = 1; i < count; i++)
      {
        sum += getDelta(selfOldValue, oldPositions[indexArray[offset + i]].position, coefficients[offset + i]);
      }
      newPositions[index].position += sum * (successiveOverRealaxation / count);
      // division is for under relaxation
      // concept of constraint averaging [Bridson et al. 2002], or masssplitting [Tonge et al. 2012].
      // SOR is from unified particle physics
    }
  }
}

Kernel void setDeltaPosition(
  Device ParticleStruct*        particleDeltas,
  const Device ParticleStruct*  particles,
  const Device ParticleStruct*  newParticles,
  const uint                    nodeCount)
{
  const uint index = threadIndex();
  if (index < nodeCount)
  {
    particleDeltas[index].position = newParticles[index].position - particles[index].position;
  }
}

#endif