#ifndef PHYSICS_SYSTEM_SHADER
#define PHYSICS_SYSTEM_SHADER

Kernel void integrate(
  Device ParticleStruct*            particles,
  Device ParticleStruct*            particlesPredicted,
  Device IdentityInfo*              particleIdentities,
  Device ParticleStruct*            particleDeltas,
  Device ParticleDifferential*      particleDiff,
  const Device ParticleSharedData*  particleSharedData,
  const Device ParticleAuxData*     particleAuxData,
  const Device PartitionInfo*       partitions,
  const Device EntityLocation*      entityLocation,
  Const uint4*                      globalOffsets,
  const float                       timeStep,
  const uint                        nodeCount)
{
  const uint index = threadIndex();
  const uint localIndex = threadLocalIndex();
  Shared float3 velocity[COMPUTE_MAX_THREADS];
  Shared float3 position[COMPUTE_MAX_THREADS];

  if (index < nodeCount)
  {
    const IdentityInfo identity = particleIdentities[index];

    const uint solverType = getSolverType(identity);
    const uint globalNodeOffset = globalOffsets[solverType].x;
    const uint globalSolverOffset = globalOffsets[solverType].z;
    const uint globalInstanceOffset = globalOffsets[solverType].y;

    ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(identity);
    nodeIdentity.entityId += globalSolverOffset;
    nodeIdentity.instanceId += globalInstanceOffset;

    const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];
    const ParticleNodeLocator nodeLocator = getNodeLocator(index, globalNodeOffset + partitions[nodeIdentity.instanceId].offset, entityLocation[nodeIdentity.entityId].node);

    const float invMass = getInvMass(&sharedData, particleAuxData, nodeLocator.commonNodeIndex);

    if (invMass) // only if movable
    {
      position[localIndex] = particlesPredicted[nodeLocator.absoluteNodeIndex].position + particleDeltas[nodeLocator.absoluteNodeIndex].position;

      velocity[localIndex] = (position[localIndex] - particles[nodeLocator.absoluteNodeIndex].position) / timeStep;
      velocity[localIndex] += constructFloat3(0.f, 6 * -0.98f, 0.f) * timeStep;
      velocity[localIndex] *= sharedData.velocityDamping;

      particles[nodeLocator.absoluteNodeIndex].position = position[localIndex];
      particlesPredicted[nodeLocator.absoluteNodeIndex].position = position[localIndex] + velocity[localIndex] * timeStep;
    }
  }
}

#endif