#ifndef COLLISION_SOLVER_SHADER
#define COLLISION_SOLVER_SHADER

Kernel void assignMortonCodeKernel(
  Device BVHLeafInfo*           bvhLeafs,
  const Device ParticleStruct*  particles,
  const uint length)
{
  const uint index = threadIndex();

  if (index < length)
  {
    BVHLeafInfo bvhLeaf;
    bvhLeaf.mortonCode = get32BitMortonCode(particles + index);
    bvhLeaf.index = index;

    bvhLeafs[index] = bvhLeaf;
  }
}

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
    ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(identity);

    const uint solverType = getSolverType(identity);
    const uint globalSolverOffset = globalOffsets[solverType].z;
    const uint globalNodeOffset = globalOffsets[solverType].x;
    const uint globalInstanceOffset = globalOffsets[solverType].y;

    nodeIdentity.entityId += globalSolverOffset;
    nodeIdentity.instanceId += globalInstanceOffset;

    const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];
    const ParticleNodeLocator nodeLocator = getNodeLocator(index, globalNodeOffset + partitions[nodeIdentity.instanceId].offset, entityLocation[nodeIdentity.entityId].node);

    const float invMass = getInvMass(&sharedData, particleAuxData, nodeLocator.commonNodeIndex);

    if (invMass) // only if movable
    {
      float dely = 0.f;
      //float rand;
      //particlesPredicted[absoluteNodeIndex].position.z += .01f * modf(10000.f * modf(particlesPredicted[absoluteNodeIndex].position.x + particlesPredicted[absoluteNodeIndex].position.y, &dely), &dely);
      if (particlesPredicted[nodeLocator.absoluteNodeIndex].position.y <= -2.f)
      {
        dely = -2.f - particlesPredicted[nodeLocator.absoluteNodeIndex].position.y;
        particles[nodeLocator.absoluteNodeIndex].position.y += dely;
        particlesPredicted[nodeLocator.absoluteNodeIndex].position.y += dely;
      }
    }
  }
}

#endif