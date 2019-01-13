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
  Const PhySystemOffsets*           globalOffsets,
  const float                       timeStep,
  const uint                        nodeCount)
{
  const uint index = threadIndex();

  float3 velocity, position;

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

    if (invMass) // only if movable
    {
      position = particlesPredicted[nodeLocator.absoluteNodeIndex].position + particleDeltas[nodeLocator.absoluteNodeIndex].position;

      velocity = (position - particles[nodeLocator.absoluteNodeIndex].position) / timeStep;
      velocity += constructFloat3(0.f, -0.98f, 0.f) * timeStep;
      velocity *= sharedData.velocityDamping;

      particles[nodeLocator.absoluteNodeIndex].position = position;
      particlesPredicted[nodeLocator.absoluteNodeIndex].position = position + velocity * timeStep;
    }
  }
}

#endif