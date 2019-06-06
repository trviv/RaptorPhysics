#ifndef DISTANCE_SOLVER_SHADER
#define DISTANCE_SOLVER_SHADER

#define successiveOverRealaxation 1.5f

/*
@function Function to calculate gradient between 2 constrained objects
@param selfOldValue Current value for object 1.
@param otherOldValue Current value for object 2.
@param coefficient Constrain magnitude.
*/
VariableType getDelta(
  const VariableType    selfOldValue,
  const VariableType    otherOldValue,
  const CoefficientType coefficient)
{
  VariableType delta = otherOldValue - selfOldValue;
  const float deltaLength = length(delta);
  if (deltaLength > COMPUTE_EPSILON)
  {
    delta *= ((1.f - (coefficient / deltaLength)) * .5f);
  }
  return delta;
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
  const uint                        nodeCount)
{
  const uint index = threadIndex();

  if (index < nodeCount)
  {
    ParticleStruct oldValue = oldPositions[index];

    const ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(oldValue.identity);
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
      oldValue.position += sum * (successiveOverRealaxation / count);
      // division is for under relaxation
      // concept of constraint averaging [Bridson et al. 2002], or masssplitting [Tonge et al. 2012].
      // SOR is from unified particle physics
    }
    newPositions[index] = oldValue;
  }
}

/*
@kernel Calculate delta position for particles.
@param particleDeltas Position delta output buffer.
@param particles Old position buffer.
@param newParticles New position buffer.
@param nodeCount Total nodes in the solver.
*/
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
