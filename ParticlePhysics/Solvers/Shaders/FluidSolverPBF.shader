#ifndef FLUID_SOLVER_PBF_SHADER
#define FLUID_SOLVER_PBF_SHADER

inline float scorrFunction(const float r, const float h)
{
  const float corrK = 0.01f;
  const float corrDelQ = 0.1f;
  //const int corrN = 4;
  const float x = poly6FunctionVariable(r, h) / poly6FunctionVariable(corrDelQ * h, h);
  //return -corrK * pow(x, corrN);
  const float t = x * x;
  return -corrK * t * t;
}

/*
@kernel Get particle lambdas.
@param particlesDensity Particle densities output.
@param particlesLambda Particle lambdas output.
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
Kernel void calculateLambda(
  Device float*                       particlesDensity,
  Device float*                       particlesLambda,
  const Device uint*                  gridCellParticleOffsets,
  const Device uint*                  gridParticleCellIndex,
  const Device ParticleStruct*        particlesPredicted,
  const Device ParticleSharedData*    particleSharedData,
  Const XAB*                          systemBoundingBox,
  Const float*                        invRadius,
  constantKernelInput(int,            gridSize),
  constantKernelInput(int,            gridSizeExp),
  constantKernelInput(uint,           nodeCount)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint particleIndex = threadIndex();

  if (particleIndex >= nodeCount)
  {
    return;
  }

  const uint gridCellIndex = gridParticleCellIndex[particleIndex];
  DECLARE_SELF_PARTICLE(particlesPredicted, identity, nodeIdentity)
  const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];
  const float mass = 1.f/particleSharedData[nodeIdentity.entityId].sharedInvMass;
  const FluidSolverData fluidSolverData = particleSharedData[nodeIdentity.entityId].fluidSolverData;

  const float density = calculateParticleDensity(selfParticle, mass, fluidSolverData, gridCellParticleOffsets, particlesPredicted, gridCellIndex, gridSize, gridSizeExp);

  particlesDensity[particleIndex] = density;
  particlesLambda[particleIndex] = -(density * sharedData.invRestDensity - 1.f) * fluidSolverData.fluidKernelFunctionConstant[2];
}

/*
@kernel Resolve particle collisions.
@param particlesNew Aligned old particle position.
@param particlesOld Old particle position.
@param particlesPredictedOld Integrated particle position.
@param particlesDensity Particle densities output.
@param particlesLambda Particle lambdas output.
@param gridCellParticleOffsets Starting offset for each grid cell.
@param gridParticleCellIndex Computed cell index for each particle.
@param particleSharedData Particle entity shared data.
@param systemBoundingBox Physics system's bounding box.
@param invRadius Inverse of max particle radius in the system.
@param gridSize Size of grid in one dimension.
@param gridSizeExp Grid size in power of 2.
@param nodeCount Total nodes in the solver.
*/
Kernel void applyCorrection(
  Device ParticleStruct*              particlesPredictedNew,
  const Device ParticleStruct*        particlesPredictedOld,
  const Device float*                 particlesDensity,
  const Device float*                 particlesLambda,
  const Device uint*                  gridCellParticleOffsets,
  const Device uint*                  gridParticleCellIndex,
  const Device ParticleSharedData*    particleSharedData,
  Const XAB*                          systemBoundingBox,
  Const float*                        invRadius,
  constantKernelInput(int,            gridSize),
  constantKernelInput(int,            gridSizeExp),
  constantKernelInput(uint,           nodeCount)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint particleIndex = threadIndex();

  if (particleIndex >= nodeCount)
  {
    return;
  }

  const uint gridCellIndex = gridParticleCellIndex[particleIndex];

  float3 delta = constructFloat3(0.f);

  // current particle data
  DECLARE_SELF_PARTICLE(particlesPredictedOld, identity, nodeIdentity)

  const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];
  const float lambda = particlesLambda[particleIndex];

#ifdef GRID_SOLVER_HASH_FUNCTION
  const float3 particleCellPosition = (selfParticle.position - systemBoundingBox->min) * invRadius[0];
#endif

  const float fluidKernelRadiusSq = sqr(sharedData.fluidSolverData.fluidKernelRadius);

  GRID_SOLVER_NEIGHBOUR_PARTICLE_LOOP_BEGIN
    const ParticleStruct otherParticle = particlesPredictedOld[otherNodeIndex];
    const float3 collisionVector = selfParticle.position - otherParticle.position;
    const float actualDistanceSq = lengthSq(collisionVector);

    if (actualDistanceSq >= fluidKernelRadiusSq)
    {
      continue;
    }

    const float actualDistance = sqrt(actualDistanceSq);
    const float scorr = scorrFunction(actualDistance, sharedData.fluidSolverData.fluidKernelRadius);
    delta += collisionVector * ((lambda + particlesLambda[otherNodeIndex] + scorr) * spikyFunctionGradientVariable(actualDistance, sharedData.fluidSolverData.fluidKernelRadius));
  GRID_SOLVER_NEIGHBOUR_PARTICLE_LOOP_END

  selfParticle.position += delta * sharedData.invRestDensity * sharedData.fluidSolverData.fluidKernelFunctionConstant[1];
  selfParticle.identity = identity;

  particlesPredictedNew[particleIndex] = selfParticle;
}

/*
@kernel Update velocities based on updated positions.
@param particlesPosition Particle positions.
@param particlesPressureForce Particle forces calculated from pressure.
@param particleSharedData Particle entity shared data.
@param partitions Instance partition data.
@param gridParticleCellIndex Computed cell index for each particle.
@param nodeCount Total nodes in the solver.
@param occupiedCellCount Total active grid cells.
*/
Kernel void updateVelocities(
  Device ParticleDifferential*        particlesDiff,
  const Device ParticleStruct*        particlesPredicted,
  const Device ParticleStruct*        particles,
  constantKernelInput(uint,           nodeCount),
  constantKernelInput(float,          timeStep)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint particleIndex = threadIndex();

  if (particleIndex >= nodeCount)
  {
    return;
  }

  particlesDiff[particleIndex].velocity = (particlesPredicted[particleIndex].position - particles[particleIndex].position) / timeStep;
}

/*
@kernel Calculate vorticity omega value.
@param particlesOmega Omega value for particles.
@param particlesDiff Particle velocity.
@param particlesPredictedOld Integrated particle position.
@param gridCellParticleOffsets Starting offset for each grid cell.
@param gridParticleCellIndex Computed cell index for each particle.
@param particleSharedData Particle entity shared data.
@param systemBoundingBox Physics system's bounding box.
@param invRadius Inverse of max particle radius in the system.
@param gridSize Size of grid in one dimension.
@param gridSizeExp Grid size in power of 2.
@param nodeCount Total nodes in the solver.
*/
Kernel void vorticityOmega(
  Device float3*                      particlesOmega,
  const Device ParticleDifferential*  particlesDiff,
  const Device ParticleStruct*        particlesPredictedOld,
  const Device uint*                  gridCellParticleOffsets,
  const Device uint*                  gridParticleCellIndex,
  const Device ParticleSharedData*    particleSharedData,
  Const XAB*                          systemBoundingBox,
  Const float*                        invRadius,
  constantKernelInput(int,            gridSize),
  constantKernelInput(int,            gridSizeExp),
  constantKernelInput(uint,           nodeCount)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint particleIndex = threadIndex();

  if (particleIndex >= nodeCount)
  {
    return;
  }

  const uint gridCellIndex = gridParticleCellIndex[particleIndex];

  float3 sumOmega = constructFloat3(0.f);

  // current particle data
  DECLARE_SELF_PARTICLE(particlesPredictedOld, identity, nodeIdentity)

  ParticleDifferential selfParticleDiff = particlesDiff[particleIndex];
  const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];

#ifdef GRID_SOLVER_HASH_FUNCTION
  const float3 particleCellPosition = (selfParticle.position - systemBoundingBox->min) * invRadius[0];
#endif

  const float fluidKernelRadiusSq = sqr(sharedData.fluidSolverData.fluidKernelRadius);

  GRID_SOLVER_NEIGHBOUR_PARTICLE_LOOP_BEGIN
    const ParticleStruct otherParticle = particlesPredictedOld[otherNodeIndex];
    const float3 collisionVector = selfParticle.position - otherParticle.position;
    const float actualDistanceSq = lengthSq(collisionVector);

    if (actualDistanceSq >= fluidKernelRadiusSq)
    {
      continue;
    }

    const float actualDistance = sqrt(actualDistanceSq);
    const float3 velocityDiff = particlesDiff[otherNodeIndex].velocity - selfParticleDiff.velocity;
    sumOmega += cross(velocityDiff, collisionVector * (spikyFunctionGradientVariable(actualDistance, sharedData.fluidSolverData.fluidKernelRadius)));
  GRID_SOLVER_NEIGHBOUR_PARTICLE_LOOP_END

  particlesOmega[particleIndex] = sumOmega * sharedData.fluidSolverData.fluidKernelFunctionConstant[1];
}

/*
@kernel Calculate vorticity confinement and XSPH viscosity for particles.
@param particleForce Force applied to the particle due to the fluid..
@param particlesPredictedOld Integrated particle position.
@param particlesDiff Particle velocity.
@param particlesOmega Omega value for particles.
@param gridCellParticleOffsets Starting offset for each grid cell.
@param gridParticleCellIndex Computed cell index for each particle.
@param particleSharedData Particle entity shared data.
@param systemBoundingBox Physics system's bounding box.
@param invRadius Inverse of max particle radius in the system.
@param gridSize Size of grid in one dimension.
@param gridSizeExp Grid size in power of 2.
@param nodeCount Total nodes in the solver.
@param timeStep Simulation time step.
*/
Kernel void vorticityConfinementXSPHViscosity(
  Device ParticleForce*               particleForce,
  const Device ParticleStruct*        particlesPredictedOld,
  const Device ParticleDifferential*  particlesDiff,
  const Device float3*                particlesOmega,
  const Device float*                 particlesDensity,
  const Device uint*                  gridCellParticleOffsets,
  const Device uint*                  gridParticleCellIndex,
  const Device ParticleSharedData*    particleSharedData,
  Const XAB*                          systemBoundingBox,
  Const float*                        invRadius,
  constantKernelInput(int,            gridSize),
  constantKernelInput(int,            gridSizeExp),
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

  float3 delta = constructFloat3(0.f);
  float3 omegaDelta = constructFloat3(0.f);

  // current particle data
  DECLARE_SELF_PARTICLE(particlesPredictedOld, identity, nodeIdentity)

  ParticleDifferential selfParticleDiff = particlesDiff[particleIndex];
  const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];
  const float3 selfOmega = particlesOmega[particleIndex];

#ifdef GRID_SOLVER_HASH_FUNCTION
  const float3 particleCellPosition = (selfParticle.position - systemBoundingBox->min) * invRadius[0];
#endif

  const float fluidKernelRadiusSq = sqr(sharedData.fluidSolverData.fluidKernelRadius);

  GRID_SOLVER_NEIGHBOUR_PARTICLE_LOOP_BEGIN
    const ParticleStruct otherParticle = particlesPredictedOld[otherNodeIndex];
    const float3 collisionVector = selfParticle.position - otherParticle.position;
    const float actualDistanceSq = lengthSq(collisionVector);

    if (actualDistanceSq >= fluidKernelRadiusSq)
    {
      continue;
    }

    const float actualDistance = sqrt(actualDistanceSq);
    const float3 velocityDiff = particlesDiff[otherNodeIndex].velocity - selfParticleDiff.velocity;
    delta += velocityDiff * poly6FunctionVariableSquares(actualDistanceSq, fluidKernelRadiusSq);
    omegaDelta += collisionVector * (length(particlesOmega[otherNodeIndex]) * spikyFunctionGradientVariable(actualDistance, sharedData.fluidSolverData.fluidKernelRadius));
  GRID_SOLVER_NEIGHBOUR_PARTICLE_LOOP_END

  delta *= sharedData.viscosity * sharedData.fluidSolverData.fluidKernelFunctionConstant[0] / (timeStep * sharedData.sharedInvMass);
  omegaDelta *= sharedData.fluidSolverData.fluidKernelFunctionConstant[1];

  const float omegaDeltaLength = length(omegaDelta);
  if (omegaDeltaLength > COMPUTE_EPSILON)
  {
    delta += cross(omegaDelta / omegaDeltaLength, selfOmega) * 0.01f / sharedData.sharedInvMass;
  }

  particleForce[particleIndex].force = delta;
}

#endif
