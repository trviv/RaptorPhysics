#ifndef PHYSICS_SYSTEM_SHADER
#define PHYSICS_SYSTEM_SHADER

Kernel void integrate(
  Device ParticleStruct*            particles,
  Device IdentityInfo*              particleIdentities,
  Device ParticleStruct*            particleDeltas,
  Device ParticleDifferential*      particleDiff,
  const Device ParticleSharedData*  particleSharedData,
  const Device ParticleAuxData*     particleAuxData,
  Const uint*                       solverEntityOffsets,
  Const uint*                       solverNodeOffsets,
  const float                       timeStep,
  const uint                        nodeCount)
{
  const uint index = threadIndex();
  const uint localIndex = threadLocalIndex();
  Shared float3 velocity[COMPUTE_MAX_THREADS];

  if (index < nodeCount)
  {
    const IdentityInfo identity = particleIdentities[index];
    const uint solverId = getSolverId(identity);
    const uint instanceId = getInstanceId(identity);
    const uint entityId = solverEntityOffsets[solverId] + getEntityId(identity);

    const ParticleSharedData sharedData = particleSharedData[entityId];
    const float invMass = getInvMass(&sharedData, particleAuxData, index);

    if (invMass) // only if movable
    {
      velocity[localIndex] = particleDeltas[index].position / timeStep;
      velocity[localIndex] += constructFloat3(0.f, -9.8f, 0.f) * timeStep;
      velocity[localIndex] *= sharedData.velocityDamping;
      particleDiff[index].velocity = velocity[localIndex];

      particles[index].position += particleDiff[index].velocity * timeStep;
      if (particles[index].position.y <= -2)
      {
        particles[index].position.y = -2;
      }
      //particles[index].position += particleDeltas[index].position;
      /*
      particleDiffs[index] += timeStep*particleSharedData[particleAuxData[index].identity].damping
      constrain->getVelocity()[index] += (constrain->del_t *
      constrain->velocity_fraction * constrain->getMass()[index]) * constrain->getForce()[index];
      constrain->getPosition()[index] += constrain->getVelocity()[index] * constrain->del_t;
      if (constrain->getPosition()[index][Y] <= -2)
      constrain->getPosition()[index][Y] = -2;
      constrain->getForce()[index] = Real3(0, -9.8, 0);*/
    }
  }
}

#endif