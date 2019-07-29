#ifndef PHYSICS_SYSTEM_SHADER
#define PHYSICS_SYSTEM_SHADER

//#define DEBUG_PHYSICS_SYSTEM

//#define USE_ALTERNATIVE_KERNEL_ARGS

Kernel void integrateDifferentiateStep(
  Device ParticleStruct*              particles,
#ifndef USE_ALTERNATIVE_KERNEL_ARGS
  Device ParticleStruct*              particlesPredicted,
  Device ParticleDifferential*        particleDiff,
#else
  const Device ParticleStruct*        particlesPredictedIn,
  const Device ParticleDifferential*  particleDiffIn,
#endif
  const Device ParticleSharedData*    particleSharedData,
  const Device ParticleAuxData*       particleAuxData,
  const Device PartitionInfo*         partitions,
  const Device EntityLocation*        entityLocation,
  Const PhySystemOffsets*             globalOffsets,
  const float                         timeStep,
  const uint                          nodeCount)
{
  const uint index = threadIndex();

#ifdef USE_ALTERNATIVE_KERNEL_ARGS
  Device ParticleStruct* particlesPredicted = particlesPredictedIn;
  Device ParticleDifferential* particleDiff = particleDiffIn;
#endif

  if (index < nodeCount)
  {
    ParticleStruct particle = particles[index];
    const IdentityInfo identity = particle.identity;
    ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(identity);

    const PhySystemOffsets phySystemOffsets = globalOffsets[nodeIdentity.solverType];

    nodeIdentity.entityId += phySystemOffsets.globalSolverOffset;
    nodeIdentity.instanceId += phySystemOffsets.globalInstanceOffset;

    const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];
    const ParticleNodeLocator nodeLocator = getNodeLocator(index, phySystemOffsets.globalNodeOffset + partitions[nodeIdentity.instanceId].offset, entityLocation[nodeIdentity.entityId].node);

    const float invMass = getInvMassUsingDeviceAux(&sharedData, particleAuxData, nodeLocator.commonNodeIndex);

    float3 velocity, particlePositionPredicted;

#ifdef DEBUG_PHYSICS_SYSTEM
    printf ("In: %d %d %f %f %f\n", index, identity.identity, particleDiff[index].velocity.x, particleDiff[index].velocity.y, particleDiff[index].velocity.z);
    printf ("In: %d %d %f %f %f\n", index, identity.identity, particlesPredicted[index].position.x, particlesPredicted[index].position.y, particlesPredicted[index].position.z);
#endif

    if (invMass) // only if movable
    {
      particlePositionPredicted = particlesPredicted[index].position;
      velocity = (particlePositionPredicted - particle.position) / timeStep;

      particle.position = particlePositionPredicted;
      particle.identity = identity;
      particles[index] = particle;

      velocity += constructFloat3(0.f, -0.98f, 0.f) * timeStep;
      velocity *= sharedData.velocityDamping;
      velocity = select(velocity, constructFloat3(0.f), fabs(velocity)<0.01f);

      particleDiff[index].velocity = velocity;
      particle.position += velocity * timeStep;
      particle.identity = identity;
      particlesPredicted[index] = particle;
    }
#ifdef DEBUG_PHYSICS_SYSTEM
    printf("Out: %d %d %f %f %f\n", index, identity.identity, particle.position.x, particle.position.y, particle.position.z);
#endif
  }
}

Kernel void startStep(
  const Device ParticleStruct*        particles,
#ifndef USE_ALTERNATIVE_KERNEL_ARGS
  Device ParticleStruct*            	particlesPredicted,
  Device ParticleDifferential*        particleDiff,
#else
  const Device ParticleStruct*        particlesPredictedIn,
  const Device ParticleDifferential*  particleDiffIn,
#endif
  const Device ParticleSharedData*    particleSharedData,
  const Device ParticleAuxData*       particleAuxData,
  const Device PartitionInfo*       	partitions,
  const Device EntityLocation*        entityLocation,
  Const PhySystemOffsets*           	globalOffsets,
  const float                       	timeStep,
  const uint                          nodeCount)
{
  const uint index = threadIndex();

#ifdef USE_ALTERNATIVE_KERNEL_ARGS
  Device ParticleStruct* particlesPredicted = particlesPredictedIn;
  Device ParticleDifferential* particleDiff = particleDiffIn;
#endif

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

#ifdef DEBUG_PHYSICS_SYSTEM
    printf ("In: %d %d %f %f %f\n", index, identity.identity, particleDiff[index].velocity.x, particleDiff[index].velocity.y, particleDiff[index].velocity.z);
    printf ("In: %d %d %f %f %f\n", index, identity.identity, particlesPredicted[index].position.x, particlesPredicted[index].position.y, particlesPredicted[index].position.z);
#endif

    if (invMass) // only if movable
    {
      velocity = particleDiff[index].velocity;
      velocity += constructFloat3(0.f, -9.8f, 0.f) * timeStep;
      velocity *= sharedData.velocityDamping;
      velocity = select(velocity, constructFloat3(0.f), fabs(velocity)<0.01f);

      particleDiff[index].velocity = velocity;
      particle.position += velocity * timeStep;
      particle.identity = identity;
      particlesPredicted[index] = particle;
    }
#ifdef DEBUG_PHYSICS_SYSTEM
    printf("Out: %d %d %f %f %f\n", index, identity.identity, particle.position.x, particle.position.y, particle.position.z);
#endif
  }
}

Kernel void endStep(
#ifndef USE_ALTERNATIVE_KERNEL_ARGS
  Device ParticleStruct*              particles,
#else
  const Device ParticleStruct*        particlesIn,
#endif
  const Device ParticleStruct*        particlesPredicted,
  Device ParticleDifferential*        particleDiff,
  const Device ParticleSharedData*    particleSharedData,
  const Device ParticleAuxData*       particleAuxData,
  const Device PartitionInfo*         partitions,
  const Device EntityLocation*        entityLocation,
  Const PhySystemOffsets*             globalOffsets,
  const float                       	timeStep,
  const uint                          nodeCount)
{
  const uint index = threadIndex();

  if (index < nodeCount)
  {
#ifdef USE_ALTERNATIVE_KERNEL_ARGS
    Device ParticleStruct* particles = particlesIn;
#endif

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
    printf ("In: %d %d %f %f %f\n", index, identity.identity, particleDiff[index].velocity.x, particleDiff[index].velocity.y, particleDiff[index].velocity.z);
    printf ("In: %d %d %f %f %f\n", index, identity.identity, particlesPredicted[index].position.x, particlesPredicted[index].position.y, particlesPredicted[index].position.z);
#endif

    if (invMass) // only if movable
    {
      particlePositionPredicted = particlesPredicted[index].position;
      velocity = (particlePositionPredicted - particle.position) / timeStep;

      particle.position = particlePositionPredicted;
      particle.identity = identity;
      particles[index] = particle;

      particleDiff[index].velocity = velocity;
    }

#ifdef DEBUG_PHYSICS_SYSTEM
    printf("Out: %d %d %f %f %f\n", index, identity.identity, particleDiff[index].velocity.x, particleDiff[index].velocity.y, particleDiff[index].velocity.z);
    printf("Out: %d %d %f %f %f\n", index, identity.identity, particle.position.x, particle.position.y, particle.position.z);
#endif
  }
}

#endif
