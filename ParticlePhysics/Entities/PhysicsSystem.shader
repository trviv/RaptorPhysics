#ifndef PHYSICS_SYSTEM_SHADER
#define PHYSICS_SYSTEM_SHADER

//#define DEBUG_PHYSICS_SYSTEM
#define PHYSICS_SYSTEM_EULER
//#define PHYSICS_SYSTEM_LEAP_FROG
//#define PHYSICS_SYSTEM_VERLET

Kernel void integrateDifferentiateStep(
  Device ParticleStruct*              particles,
  Device ParticleStruct*              particlesPredicted,
  Device ParticleDifferential*        particleDiff,
  Device ParticleForce*               particleForce,
  const Device ParticleSharedData*    particleSharedData,
  const Device ParticleAuxData*       particleAuxData,
  const Device PartitionInfo*         partitions,
  const Device EntityLocation*        entityLocation,
  Const PhySystemSettings*            systemSettings,
  constantKernelInput(float,          timeStep),
  constantKernelInput(uint,           nodeCount)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint index = threadIndex();

  if (index < nodeCount)
  {
    ParticleStruct particle = particles[index];
    const IdentityInfo identity = particle.identity;
    ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(identity);

    const PhySystemOffsets phySystemOffsets = systemSettings->globalOffsets[nodeIdentity.solverType];

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

      velocity += (systemSettings->gravity + particleForce[index].force * invMass) * timeStep;
      velocity *= sharedData.collisionSolverData.velocityDamping;
      velocity = select(velocity, constructFloat3(0.f), fabs(velocity)<0.00001f);

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
  Device ParticleStruct*            	particlesPredicted,
  Device ParticleDifferential*        particleDiff,
  const Device ParticleSharedData*    particleSharedData,
  const Device ParticleAuxData*       particleAuxData,
  const Device PartitionInfo*         partitions,
  const Device EntityLocation*        entityLocation,
  Const PhySystemSettings*            systemSettings,
  constantKernelInput(float,          timeStep),
  constantKernelInput(uint,           nodeCount)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint index = threadIndex();

  if (index < nodeCount)
  {
    ParticleStruct particle = particles[index];
    IdentityInfo identity = particle.identity;
    ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(identity);

    const PhySystemOffsets phySystemOffsets = systemSettings->globalOffsets[nodeIdentity.solverType];

    nodeIdentity.entityId += phySystemOffsets.globalSolverOffset;
    nodeIdentity.instanceId += phySystemOffsets.globalInstanceOffset;

    const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];
    const ParticleNodeLocator nodeLocator = getNodeLocator(index, phySystemOffsets.globalNodeOffset + partitions[nodeIdentity.instanceId].offset, entityLocation[nodeIdentity.entityId].node);

    const float invMass = getInvMassUsingDeviceAux(&sharedData, particleAuxData, nodeLocator.commonNodeIndex);

    float3 velocity;

#ifdef DEBUG_PHYSICS_SYSTEM
    printf ("Start In: %d %d %f %f %f\n", index, identity.identity, particleDiff[index].velocity.x, particleDiff[index].velocity.y, particleDiff[index].velocity.z);
    printf ("Start In: %d %d %f %f %f\n", index, identity.identity, particlesPredicted[index].position.x, particlesPredicted[index].position.y, particlesPredicted[index].position.z);
#endif

    if (invMass) // only if movable
    {
#ifdef PHYSICS_SYSTEM_EULER
      velocity = particleDiff[index].velocity;
      velocity += systemSettings->gravity * timeStep;
      velocity *= sharedData.collisionSolverData.velocityDamping;
      velocity = select(velocity, constructFloat3(0.f), fabs(velocity) < constructFloat3(0.01f));

      particleDiff[index].velocity = velocity;
      particle.position += velocity * timeStep;
      particle.identity = identity;
      particlesPredicted[index] = particle;
#endif

#ifdef PHYSICS_SYSTEM_LEAP_FROG
      velocity = particleDiff[index].velocity + systemSettings->gravity * (timeStep * 0.5f);

      particle.position += velocity * timeStep;
      particle.identity = identity;

      velocity += systemSettings->gravity * (timeStep * 0.5f);
      velocity = select(velocity, constructFloat3(0.f), fabs(velocity) < constructFloat3(0.01f));

      particleDiff[index].velocity = velocity;
      particlesPredicted[index] = particle;
#endif

#ifdef PHYSICS_SYSTEM_VERLET
      velocity = particleDiff[index].velocity + systemSettings->gravity * (timeStep * timeStep);

      particle.position = velocity;
      particle.identity = identity;

      particleDiff[index].velocity = velocity;
      particlesPredicted[index] = particle;
#endif
    }
#ifdef DEBUG_PHYSICS_SYSTEM
    printf("Start Out: %d %d %f %f %f\n", index, identity.identity, particle.position.x, particle.position.y, particle.position.z);
#endif
  }
}

Kernel void endStep(
  Device ParticleStruct*              particles,
  const Device ParticleStruct*        particlesPredicted,
  Device ParticleDifferential*        particleDiff,
  Device ParticleForce*               particleForce,
  const Device ParticleSharedData*    particleSharedData,
  const Device ParticleAuxData*       particleAuxData,
  const Device PartitionInfo*         partitions,
  const Device EntityLocation*        entityLocation,
  Const PhySystemSettings*            systemSettings,
  constantKernelInput(float,          timeStep),
  constantKernelInput(uint,           nodeCount)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint index = threadIndex();

  if (index < nodeCount)
  {
    ParticleStruct particle = particles[index];
    IdentityInfo identity = particle.identity;
    ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(identity);

    const PhySystemOffsets phySystemOffsets = systemSettings->globalOffsets[nodeIdentity.solverType];

    nodeIdentity.entityId += phySystemOffsets.globalSolverOffset;
    nodeIdentity.instanceId += phySystemOffsets.globalInstanceOffset;

    const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];
    const ParticleNodeLocator nodeLocator = getNodeLocator(index, phySystemOffsets.globalNodeOffset + partitions[nodeIdentity.instanceId].offset, entityLocation[nodeIdentity.entityId].node);

    const float invMass = getInvMassUsingDeviceAux(&sharedData, particleAuxData, nodeLocator.commonNodeIndex);

    float3 velocity, particlePositionPredicted;

#ifdef DEBUG_PHYSICS_SYSTEM
    printf ("End In: %d %d %f %f %f\n", index, identity.identity, particleDiff[index].velocity.x, particleDiff[index].velocity.y, particleDiff[index].velocity.z);
    printf ("End In: %d %d %f %f %f\n", index, identity.identity, particlesPredicted[index].position.x, particlesPredicted[index].position.y, particlesPredicted[index].position.z);
#endif

    if (invMass) // only if movable
    {
      particlePositionPredicted = particlesPredicted[index].position;
      velocity = (particlePositionPredicted - particle.position) / timeStep;

#ifdef PHYSICS_SYSTEM_VERLET
      velocity = (2.f * particlePositionPredicted - particle.position);
#endif

      velocity += particleForce[index].force * invMass * timeStep;

      particle.position = particlePositionPredicted;
      particle.identity = identity;
      particles[index] = particle;

      particleDiff[index].velocity = velocity;
    }

#ifdef DEBUG_PHYSICS_SYSTEM
    printf("End Out: %d %d %f %f %f\n", index, identity.identity, particleDiff[index].velocity.x, particleDiff[index].velocity.y, particleDiff[index].velocity.z);
    printf("End Out: %d %d %f %f %f\n", index, identity.identity, particle.position.x, particle.position.y, particle.position.z);
#endif
  }
}

#endif
