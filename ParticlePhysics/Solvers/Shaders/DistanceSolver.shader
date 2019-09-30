#ifndef DISTANCE_SOLVER_SHADER
#define DISTANCE_SOLVER_SHADER

#define successiveOverRealaxation 1.5f

/*
@function Function to calculate gradient between 2 constrained objects
@param selfOldValue Current value for object 1.
@param otherOldValue Current value for object 2.
@param coefficient Constrain magnitude.
*/
inline VariableType getDelta(
  const VariableType    selfOldValue,
  const VariableType    otherOldValue,
  const CoefficientType coefficient)
{
  const VariableType delta = otherOldValue - selfOldValue;
  float deltaLength = length(delta);
  deltaLength = select(deltaLength, COMPUTE_EPSILON, deltaLength < COMPUTE_EPSILON);
  return delta * ((deltaLength - coefficient) * .5f / deltaLength);
}

/*
@kernel Solve distance constrain using spring equation.
@param newPositions Position output buffer for this iteration.
@param oldPositions Position input buffer for this iteration.
@param constrainNodes Buffer containing constrain header data.
@param indexArray Buffer containing constrain index data.
@param coefficients Buffer containing constrain magnitude data.
@param entityLocation Buffer containing entity boundary info.
@param nodeCount Total nodes in the solver.
*/
Kernel void distanceSolverSpring(
  Device ParticleStruct*            newPositions,
  const Device ParticleStruct*      oldPositions,
  const Device ConstrainStruct*     constrainNodes,
  const Device IndexType*           indexArray,
  const Device CoefficientType*     coefficients,
  const Device PartitionInfo*       partitions,
  const Device EntityLocation*      entityLocation,
  constantKernelInput(uint,         nodeCount)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint index = threadIndex();

  if (index < nodeCount)
  {
    ParticleStruct oldValue = oldPositions[index];
    const IdentityInfo identity = oldValue.identity;

    const ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(identity);
    const EntityLocation localEntityLocation = entityLocation[nodeIdentity.entityId];
    const ParticleNodeLocator nodeLocator = getNodeLocator(index, partitions[nodeIdentity.instanceId].offset, localEntityLocation.node);

    VariableType sum = 0;

    const ConstrainStruct constrain = constrainNodes[nodeLocator.commonNodeIndex];
    const uint commonConnectionIndex = localEntityLocation.connection.offset + constrainOffset(constrain);
    const uint count = constrainCount(constrain);

    if (coefficients[commonConnectionIndex]) // if self movement allowed
    {
      for (uint i = 1; i < count; i++)
      {
        const uint absoluteConnectionNodeIndex = nodeLocator.absoluteNodeOffset + indexArray[commonConnectionIndex + i];

        sum += getDelta(oldValue.position,
          oldPositions[absoluteConnectionNodeIndex].position,
          coefficients[commonConnectionIndex + i]);
      }
      oldValue.position += sum * (successiveOverRealaxation / (count - 1));
      // division is for under relaxation
      // concept of constraint averaging [Bridson et al. 2002], or masssplitting [Tonge et al. 2012].
      // SOR is from unified particle physics
    }
    oldValue.identity = identity;
    newPositions[index] = oldValue;
  }
}

#endif
