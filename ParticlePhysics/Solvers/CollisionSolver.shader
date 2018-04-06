#ifndef COLLISION_SOLVER_SHADER
#define COLLISION_SOLVER_SHADER

/*
@kernel Apply boundary constrain.
@param particles Initial particle position.
@param particlesPredicted Integrated particle position.
@param particleIdentities Particle identifiers.
@param particleSharedData Particle entity shared data.
@param particleAuxData Additional particle data.
@param partitions Instance partition data.
@param entityLocation Buffer containing entity boundary info.
@param nodeCount Total nodes in the solver.
*/
Kernel void boundaryCollisionKernel(
  Device ParticleStruct*            particles,
  Device ParticleStruct*            particlesPredicted,
  const Device IdentityInfo*        particleIdentities,
  const Device ParticleSharedData*  particleSharedData,
  const Device ParticleAuxData*     particleAuxData,
  const Device PartitionInfo*       partitions,
  const Device EntityLocation*      entityLocation,
  Const uint4*                      globalOffsets,
  const uint                        nodeCount)
{
  const uint index = threadIndex();
  const uint localIndex = threadLocalIndex();

  if (index < nodeCount)
  {
    const IdentityInfo identity = particleIdentities[index];

    const uint solverType = getSolverType(identity);
    const uint globalSolverOffset = globalOffsets[solverType].z;
    const uint globalNodeOffset = globalOffsets[solverType].x;
    const uint globalInstanceOffset = globalOffsets[solverType].y;

    const uint entityId = globalSolverOffset + getEntityId(identity);
    const uint instanceId = globalInstanceOffset + getInstanceId(identity);

    const ParticleSharedData sharedData = particleSharedData[entityId];

    const uint absoluteNodeOffset = globalNodeOffset + partitions[instanceId].offset;
    const uint relativeNodeIndex = index % entityLocation[entityId].node.count;
    const uint absoluteNodeIndex = absoluteNodeOffset + relativeNodeIndex;

    const float invMass = getInvMass(&sharedData, particleAuxData, entityLocation[entityId].node.offset + relativeNodeIndex);

    if (invMass) // only if movable
    {
      float dely = 0.f;

      if (particlesPredicted[absoluteNodeIndex].position.y <= -2.f)
      {
        dely = -2.f - particlesPredicted[absoluteNodeIndex].position.y;
        particles[absoluteNodeIndex].position.y += dely;
        particlesPredicted[absoluteNodeIndex].position.y += dely;
      }
    }
  }
}

#endif