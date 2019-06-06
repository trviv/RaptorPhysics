#ifndef PHYSICS_SYSTEM_SHADER
#define PHYSICS_SYSTEM_SHADER

//#define DEBUG_PHYSICS_SYSTEM

Kernel void startStep(
  Device ParticleStruct*            particles,
  Device ParticleStruct*            particlesPredicted,
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
    ParticleStruct particle = particles[index];
    IdentityInfo identity = particle.identity;
    ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(identity);

    const PhySystemOffsets phySystemOffsets = globalOffsets[nodeIdentity.solverType];

    nodeIdentity.entityId += phySystemOffsets.globalSolverOffset;
    nodeIdentity.instanceId += phySystemOffsets.globalInstanceOffset;

    const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];
    const ParticleNodeLocator nodeLocator = getNodeLocator(index, phySystemOffsets.globalNodeOffset + partitions[nodeIdentity.instanceId].offset, entityLocation[nodeIdentity.entityId].node);

    const float invMass = getInvMassUsingDeviceAux(&sharedData, particleAuxData, nodeLocator.commonNodeIndex);

    float3 velocity;

    if (invMass) // only if movable
    {
      velocity = particleDiff[index].velocity;
      velocity += constructFloat3(0.f, -0.98f, 0.f) * timeStep;
      velocity *= sharedData.velocityDamping;

      particleDiff[index].velocity = velocity;
      particle.position += velocity * timeStep;
      particle.identity = identity;
      particlesPredicted[index] = particle;
    }
  }
}

Kernel void endStep(
  Device ParticleStruct*            particles,
  Device ParticleStruct*            particlesPredicted,
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
    ParticleStruct particle = particles[index];
    IdentityInfo identity = particle.identity;
    ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(identity);

    const PhySystemOffsets phySystemOffsets = globalOffsets[nodeIdentity.solverType];

    nodeIdentity.entityId += phySystemOffsets.globalSolverOffset;
    nodeIdentity.instanceId += phySystemOffsets.globalInstanceOffset;

    const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];
    const ParticleNodeLocator nodeLocator = getNodeLocator(index, phySystemOffsets.globalNodeOffset + partitions[nodeIdentity.instanceId].offset, entityLocation[nodeIdentity.entityId].node);

    const float invMass = getInvMassUsingDeviceAux(&sharedData, particleAuxData, nodeLocator.commonNodeIndex);

    float3 velocity, particlePosition, particlePositionPredicted;

#ifdef DEBUG_PHYSICS_SYSTEM
    printf ("In: %d %f %f %f\n", index, particleDiff[index].velocity.x, particleDiff[index].velocity.y, particleDiff[index].velocity.z);
    printf ("In: %d %f %f %f\n", index, particleDeltas[nodeLocator.absoluteNodeIndex].position.x, particleDeltas[nodeLocator.absoluteNodeIndex].position.y, particleDeltas[nodeLocator.absoluteNodeIndex].position.z);
    printf ("In: %d %f %f %f\n", index, particlesPredicted[index].position.x, particlesPredicted[index].position.y, particlesPredicted[index].position.z);
#endif

    if (invMass) // only if movable
    {
      particlePositionPredicted = particlesPredicted[index].position + particleDeltas[nodeLocator.absoluteNodeIndex].position;
      velocity = (particlePositionPredicted - particle.position) / timeStep;

      particle.position = particlePositionPredicted;
      particle.identity = identity;
      particles[index] = particle;

      particleDiff[index].velocity = velocity;
    }

#ifdef DEBUG_PHYSICS_SYSTEM
    printf("Out: %d %f %f %f\n", index, particleDiff[index].velocity.x, particleDiff[index].velocity.y, particleDiff[index].velocity.z);
    printf("Out: %d %f %f %f\n", index, particle.position.x, particle.position.y, particle.position.z);
#endif
  }
}

#endif
