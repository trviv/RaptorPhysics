#ifndef FLUID_SOLVER_PCISPH_SHADER
#define FLUID_SOLVER_PCISPH_SHADER

/*
@kernel Predict position and velocity for the next step.
@param particlesNextPosition Next particle positions.
@param particlesNextVelocity Next particle velocities.
@param particlesPosition Integrated particle position.
@param nodeCount Total nodes in the solver.
*/
Kernel void predictionStep(
  Device ParticleStruct*              particlesNextPosition,
  Device ParticleDifferential*        particlesNextVelocity,
  const Device ParticleStruct*        particlesPosition,
  const Device ParticleDifferential*  particlesDiff,
  const Device ParticleForce*         particlesPressureForce,
  const Device ParticleSharedData*    particleSharedData,
  constantKernelInput(uint,           nodeCount),
  constantKernelInput(float,          timeStep)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint particleIndex = threadIndex();

  if (particleIndex >= nodeCount)
  {
    return;
  }

  // current particle data
  DECLARE_SELF_PARTICLE(particlesPosition, identity, nodeIdentity)

  const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];

  float3 velocity = particlesDiff[particleIndex].velocity + particlesPressureForce[particleIndex].force * timeStep * sharedData.sharedInvMass;
  particlesNextVelocity[particleIndex].velocity = velocity;

  selfParticle.position += velocity * timeStep;
  selfParticle.identity = identity;
  particlesNextPosition[particleIndex] = selfParticle;
}

/*
@kernel Get particle pressures.
@param particlesPressure Particle pressures output.
@param particlesDensity Particle densities.
@param gridCellParticleOffsets Starting offset for each grid cell.
@param gridParticleCellIndex Computed cell index for each particle.
@param particlesPredictedOld Integrated particle position.
@param particleSharedData Particle entity shared data.
@param systemBoundingBox Physics system's bounding box.
@param invRadius Inverse of max particle radius in the system.
@param gridSize Size of grid in one dimension.
@param gridSizeExp Grid size in power of 2.
@param nodeCount Total nodes in the solver.
*/
Kernel void calculatePressure(
  Device float*                     particlesPressure,
  const Device float*               particlesDensity,
  const Device uint*                gridCellParticleOffsets,
  const Device uint*                gridParticleCellIndex,
  const Device ParticleStruct*      particlesPredicted,
  const Device ParticleSharedData*  particleSharedData,
  Const XAB*                        systemBoundingBox,
  Const float*                      invRadius,
  constantKernelInput(ushort,       gridSize),
  constantKernelInput(ushort,       gridSizeExp),
  constantKernelInput(uint,         nodeCount),
  constantKernelInput(float,        timeStep),
  constantKernelInput(float,        invBeta)
  KERNEL_GLOBAL_ARGUMENTS)
{

  const uint particleIndex = threadIndex();

  if (particleIndex >= nodeCount)
  {
    return;
  }

  const uint gridCellIndex = gridParticleCellIndex[particleIndex];

  // current particle data
  float sumGradientMagnitude = 0.f;
  float3 sumGradientVector = constructFloat3(0.f);

  DECLARE_SELF_PARTICLE(particlesPredicted, identity, nodeIdentity)

  const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];

#ifdef GRID_SOLVER_HASH_FUNCTION
  const float3 particleCellPosition = (selfParticle.position - systemBoundingBox->min) * invRadius[0];
#endif

  const float fluidKernelRadiusSq = sqr(sharedData.fluidSolverData.fluidKernelRadius);

  GRID_SOLVER_NEIGHBOUR_PARTICLE_LOOP_BEGIN
    const ParticleStruct otherParticle = particlesPredicted[otherNodeIndex];
    const float3 collisionVector = selfParticle.position - otherParticle.position;
    const float actualDistanceSq = lengthSq(collisionVector);

    if (actualDistanceSq >= fluidKernelRadiusSq)
    {
      continue;
    }

    const float actualDistance = sqrt(actualDistanceSq);
    const float3 gradient = collisionVector * spikyFunctionGradientVariable(actualDistance, sharedData.fluidSolverData.fluidKernelRadius);
    sumGradientMagnitude += lengthSq(gradient);
    sumGradientVector += gradient;
  GRID_SOLVER_NEIGHBOUR_PARTICLE_LOOP_END

  sumGradientVector *= sharedData.fluidSolverData.fluidKernelFunctionConstant[1];
  sumGradientMagnitude *= (sharedData.fluidSolverData.fluidKernelFunctionConstant[1] * sharedData.fluidSolverData.fluidKernelFunctionConstant[1]);

  const float invSigma = -(invBeta * sharedData.fluidSolverData.fluidKernelFunctionConstant[2]);
  particlesPressure[particleIndex] += -(particlesDensity[particleIndex] - 1.f/sharedData.invRestDensity) * invSigma;
}

/*
@kernel Calculate particle forces.
@param particlesPredictedNew Updated particle positions post collision processing.
@param particlesDensity Particle densities output.
@param particlesPressure Particle pressures output.
@param gridCellParticleOffsets Starting offset for each grid cell.
@param particlesPosition Integrated particle position.
@param particleSharedData Particle entity shared data.
@param gridParticleCellIndex Computed cell index for each particle.
@param nodeCount Total nodes in the solver.
@param occupiedCellCount Total active grid cells.
*/
Kernel void calculateForces(
  Device ParticleForce*               particlesPressureForce,
  const Device float*                 particlesDensity,
  const Device float*                 particlesPressure,
  const Device uint*                  gridCellParticleOffsets,
  const Device uint*                  gridParticleCellIndex,
  const Device ParticleStruct*        particlesPosition,
  const Device ParticleSharedData*    particleSharedData,
  Const XAB*                          systemBoundingBox,
  Const float*                        invRadius,
  constantKernelInput(ushort,         gridSize),
  constantKernelInput(ushort,         gridSizeExp),
  constantKernelInput(uint,           nodeCount),
  constantKernelInput(float,          timeStep)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint particleIndex = threadIndex();

  if (particleIndex >= nodeCount)
  {
    return;
  }

  const uint gridCellIndex = gridParticleCellIndex[particleIndex];

  float3 force = constructFloat3(0.f);

  // current particle data
  DECLARE_SELF_PARTICLE(particlesPosition, identity, nodeIdentity)

  const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];
  const float selfDensity = particlesDensity[particleIndex];
  const float selfPressure = particlesPressure[particleIndex];

#ifdef GRID_SOLVER_HASH_FUNCTION
  const float3 particleCellPosition = (selfParticle.position - systemBoundingBox->min) * invRadius[0];
#endif

  const float selfPressureByDensity = selfPressure/sqr(selfDensity);
  const float fluidKernelRadiusSq = sqr(sharedData.fluidSolverData.fluidKernelRadius);

  GRID_SOLVER_PACKED_NEIGHBOUR_PARTICLE_LOOP_BEGIN
    const ParticleStruct otherParticle = particlesPosition[otherNodeIndex];
    const float3 collisionVector = selfParticle.position - otherParticle.position;
    const float actualDistanceSq = lengthSq(collisionVector);

    if (actualDistanceSq >= fluidKernelRadiusSq)
    {
      continue;
    }

    const float actualDistance = sqrt(actualDistanceSq);
    const float density = selfPressureByDensity + particlesPressure[otherNodeIndex]/sqr(particlesDensity[otherNodeIndex]);
    force += collisionVector * (density * spikyFunctionGradientVariable(actualDistance, sharedData.fluidSolverData.fluidKernelRadius));
  GRID_SOLVER_NEIGHBOUR_PARTICLE_LOOP_END

  particlesPressureForce[particleIndex].force = -(force * sharedData.fluidSolverData.fluidKernelFunctionConstant[1]) / sqr(sharedData.sharedInvMass);
}

#endif
