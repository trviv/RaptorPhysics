#ifndef PHYSICS_SYSTEM_SHADER
#define PHYSICS_SYSTEM_SHADER

Kernel void integrate(
  Device ParticleStruct*            particles,
  Device ParticleStruct*            particleDeltas,
  Device ParticleDifferential*      particleDiff,
  const Device ParticleSharedData*  particleSharedData,
  const Device ParticleAuxData*     particleAuxData,
  const float                       timeStep,
  const uint                        nodeCount)
{
  const uint index = threadIndex();

  if (index < nodeCount)
  {
    const ParticleSharedData sharedData = particleSharedData[particles[index].identity];
    const float invMass = getInvMass(&sharedData, particleAuxData, index);

    if (invMass)
    {
      particleDiff[index].velocity = particleDeltas[index].position / timeStep;
      particleDiff[index].velocity += constructFloat3(0.f, -9.8f, 0.f) * timeStep;
      particleDiff[index].velocity *= sharedData.velocityDamping;

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