#ifndef PHYSICS_SYSTEM_SHADER
#define PHYSICS_SYSTEM_SHADER

Kernel void startStep(
  Device ParticleStruct*            particles,
  Device ParticleStruct*            particlesPredicted,
  Device IdentityInfo*              particleIdentities,
  Device ParticleStruct*            particleDeltas,
  Device ParticleDifferential*      particleDiff,
  const Device ParticleSharedData*  particleSharedData,
  const Device ParticleAuxData*     particleAuxData,
  const Device PartitionInfo*       partitions,
  const Device EntityLocation*      entityLocation,
  Const PhySystemOffsets*           globalOffsets,
  const float                       timeStep,
  const uint                        nodeCount)
{
  const uint index = threadIndex();

  if (index < nodeCount)
  {
    const IdentityInfo identity = particleIdentities[index];
    ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(identity);

    const PhySystemOffsets phySystemOffsets = globalOffsets[nodeIdentity.solverType];

    nodeIdentity.entityId += phySystemOffsets.globalSolverOffset;
    nodeIdentity.instanceId += phySystemOffsets.globalInstanceOffset;

    const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];
    const ParticleNodeLocator nodeLocator = getNodeLocator(index, phySystemOffsets.globalNodeOffset + partitions[nodeIdentity.instanceId].offset, entityLocation[nodeIdentity.entityId].node);

    const float invMass = getInvMassUsingDeviceAux(&sharedData, particleAuxData, nodeLocator.commonNodeIndex);

    float3 velocity, particlePosition;

    if (invMass) // only if movable
    {
      velocity = particleDiff[nodeLocator.absoluteNodeIndex].velocity;
      velocity += constructFloat3(0.f, -0.98f, 0.f) * timeStep;
      velocity *= sharedData.velocityDamping;

      particleDiff[nodeLocator.absoluteNodeIndex].velocity = velocity;
      particlePosition = particles[nodeLocator.absoluteNodeIndex].position;
      particlesPredicted[nodeLocator.absoluteNodeIndex].position = particlePosition + velocity * timeStep;
    }
  }
}

Kernel void endStep(
  Device ParticleStruct*            particles,
  Device ParticleStruct*            particlesPredicted,
  Device IdentityInfo*              particleIdentities,
  Device ParticleStruct*            particleDeltas,
  Device ParticleDifferential*      particleDiff,
  const Device ParticleSharedData*  particleSharedData,
  const Device ParticleAuxData*     particleAuxData,
  const Device PartitionInfo*       partitions,
  const Device EntityLocation*      entityLocation,
  Const PhySystemOffsets*           globalOffsets,
  const float                       timeStep,
  const uint                        nodeCount)
{
  const uint index = threadIndex();

  if (index < nodeCount)
  {
    const IdentityInfo identity = particleIdentities[index];
    ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(identity);

    const PhySystemOffsets phySystemOffsets = globalOffsets[nodeIdentity.solverType];

    nodeIdentity.entityId += phySystemOffsets.globalSolverOffset;
    nodeIdentity.instanceId += phySystemOffsets.globalInstanceOffset;

    const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];
    const ParticleNodeLocator nodeLocator = getNodeLocator(index, phySystemOffsets.globalNodeOffset + partitions[nodeIdentity.instanceId].offset, entityLocation[nodeIdentity.entityId].node);

    const float invMass = getInvMassUsingDeviceAux(&sharedData, particleAuxData, nodeLocator.commonNodeIndex);

    float3 velocity, particlePosition, particlePositionPredicted;

    if (invMass) // only if movable
    {
      particlePositionPredicted = particlesPredicted[nodeLocator.absoluteNodeIndex].position + particleDeltas[nodeLocator.absoluteNodeIndex].position;
      particlePosition = particles[nodeLocator.absoluteNodeIndex].position;
      particles[nodeLocator.absoluteNodeIndex].position = particlePositionPredicted;

      particleDiff[nodeLocator.absoluteNodeIndex].velocity = (particlePositionPredicted - particlePosition) / timeStep;
    }
  }
}

#endif