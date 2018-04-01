#ifndef PHYSICS_SYSTEM_SHADER
#define PHYSICS_SYSTEM_SHADER

Kernel void integrate(
  Device ParticleStruct*            particles,
  Device IdentityInfo*              particleIdentities,
  Device ParticleStruct*            particleDeltas,
  Device ParticleDifferential*      particleDiff,
  const Device ParticleSharedData*  particleSharedData,
  const Device ParticleAuxData*     particleAuxData,
  const Device PartitionInfo*       partitions,
  const Device SectionData*         sectionData,
  Const uint4*                      globalOffsets,
  const float                       timeStep,
  const uint                        nodeCount)
{
  const uint index = threadIndex();
  const uint localIndex = threadLocalIndex();
  Shared float3 velocity[COMPUTE_MAX_THREADS];

  if (index < nodeCount)
  {
    const IdentityInfo identity = particleIdentities[index];

    const uint solverType = getSolverType(identity);
    const uint globalSolverOffset = globalOffsets[solverType].z;
    const uint globalNodeOffset = globalOffsets[solverType].x;
    const uint globalInstanceOffset = globalOffsets[solverType].y;

    const uint solverId = globalSolverOffset + getSolverId(identity);
    const uint entityId = globalInstanceOffset + getEntityId(identity);

    const ParticleSharedData sharedData = particleSharedData[solverId];

    const uint absoluteNodeOffset = globalNodeOffset + partitions[entityId].offset;
    const uint relativeNodeIndex = index % sectionData[solverId].node.count;
    const uint absoluteNodeIndex = absoluteNodeOffset + relativeNodeIndex;

    const float invMass = getInvMass(&sharedData, particleAuxData, sectionData[solverId].node.offset + relativeNodeIndex);

    if (invMass) // only if movable
    {
      velocity[localIndex] = particleDeltas[absoluteNodeIndex].position / timeStep;
      velocity[localIndex] += constructFloat3(0.f, -9.8f, 0.f) * timeStep;
      velocity[localIndex] *= sharedData.velocityDamping;

      particleDiff[absoluteNodeIndex].velocity = velocity[localIndex];
      particles[absoluteNodeIndex].position += velocity[localIndex] * timeStep;

      if (particles[absoluteNodeIndex].position.y <= -2)
      {
        particles[absoluteNodeIndex].position.y = -2;
      }
    }

    /*
    particleDiffs[index] += timeStep*particleSharedData[particleAuxData[index].identity].damping
    constrain->getVelocity()[index] += (constrain->del_t *
    constrain->velocity_fraction * constrain->getMass()[index]) * constrain->getForce()[index];
    constrain->getPosition()[index] += constrain->getVelocity()[index] * constrain->del_t;
    if (constrain->getPosition()[index][Y] <= -2)
    constrain->getPosition()[index][Y] = -2;
    constrain->getForce()[index] = Real3(0, -9.8, 0);*/
    //}
  }
}

#endif