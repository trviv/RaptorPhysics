#ifndef FLUID_SOLVER_PBF_SHADER
#define FLUID_SOLVER_PBF_SHADER

inline float scorrFunction(const float r, const float h)
{
  const float corrK = 0.1f;
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
@param gridCellParticleIndices Output array for particle indices.
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
  const Device uint*                  gridCellParticleIndices,
  const Device uint*                  gridParticleCellIndex,
  const Device ParticleStruct*        particlesPredictedOld,
  const Device ParticleSharedData*    particleSharedData,
  Const XAB*                          systemBoundingBox,
  Const float*                        invRadius,
  constantKernelInput(int,            gridSize),
  constantKernelInput(int,            gridSizeExp),
  constantKernelInput(uint,           nodeCount)
  KERNEL_GLOBAL_ARGUMENTS
  KERNEL_THREAD_ARGUMENTS
  KERNEL_THREADGROUP_ARGUMENTS)
{
  uint particleIndex = threadIndex();

  if (particleIndex >= nodeCount)
  {
    return;
  }

  particleIndex = gridCellParticleIndices[particleIndex];

  const uint gridCellIndex = gridParticleCellIndex[particleIndex];

  // current particle data
  float density = 0.f;
  float sumGradientMagnitude = 0.f;
  float3 sumGradientVector = constructFloat3(0.f);

  const ParticleStruct selfParticle = particlesPredictedOld[particleIndex];
  const IdentityInfo identity = selfParticle.identity;
  const ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(identity);
  const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];

  const short3 particleGridCellIndex = constructShort3(
    gridCellIndex & (gridSize - 1),
    (gridCellIndex >> gridSizeExp) & (gridSize - 1),
    gridCellIndex >> (gridSizeExp << 1)
  );

#ifdef GRID_SOLVER_HASH_FUNCTION
  const float3 particleCellPosition = (selfParticle.position - systemBoundingBox->min) * invRadius[0];
#endif

  // loop through neighboring cells
  GRID_SOLVER_NEIGHBOUR_LOOP_BEGIN
    const uint2 indexRange = getRangeFromOffset(gridCellParticleOffsets, gridCellIndex);

    // iterate over particles in the cell
    for (int otherParticlePointerIndex = indexRange.x; otherParticlePointerIndex < indexRange.y; otherParticlePointerIndex++)
    {
      const int otherNodeIndex = gridCellParticleIndices[otherParticlePointerIndex];
      const ParticleStruct otherParticle = particlesPredictedOld[otherNodeIndex];
      const float3 collisionVector = selfParticle.position - otherParticle.position;
      const float actualDistance = length(collisionVector);

      density += select(0.f, poly6FunctionVariable(actualDistance, sharedData.fluidKernelRadius), actualDistance < sharedData.fluidKernelRadius);

      if (actualDistance >= sharedData.fluidKernelRadius || actualDistance <= COMPUTE_EPSILON || otherNodeIndex == particleIndex)
      {
        continue;
      }

      const float3 gradient = collisionVector * spikyFunctionGradientVariable(actualDistance, sharedData.fluidKernelRadius);
      sumGradientMagnitude += lengthSq(gradient);
      sumGradientVector += gradient;
    }
  GRID_SOLVER_NEIGHBOUR_LOOP_END

  sumGradientMagnitude += lengthSq(sumGradientVector);
  sumGradientMagnitude *= (sharedData.fluidKernelFunctionConstant[1] * sharedData.fluidKernelFunctionConstant[1]);

  density *= sharedData.fluidKernelFunctionConstant[0] / sharedData.sharedInvMass;

  particlesDensity[particleIndex] = density;

  density *= sharedData.invRestDensity;

  particlesLambda[particleIndex] = -select((density - 1.0f)/(sumGradientMagnitude + COMPUTE_EPSILON), 0.f, density < 1.f);
}

/*
@kernel Resolve particle collisions.
@param particlesNew Aligned old particle position.
@param particlesOld Old particle position.
@param particlesPredictedOld Integrated particle position.
@param particlesDensity Particle densities output.
@param particlesLambda Particle lambdas output.
@param gridCellParticleOffsets Starting offset for each grid cell.
@param gridCellParticleIndices Output array for particle indices.
@param gridParticleCellIndex Computed cell index for each particle.
@param particleSharedData Particle entity shared data.
@param systemBoundingBox Physics system's bounding box.
@param invRadius Inverse of max particle radius in the system.
@param gridSize Size of grid in one dimension.
@param gridSizeExp Grid size in power of 2.
@param nodeCount Total nodes in the solver.
*/
Kernel void calculateForces(
  Device ParticleStruct*              particlesNew,
  const Device ParticleStruct*        particlesOld,
  Device ParticleStruct*              particlesPredictedOld,
  const Device float*                 particlesDensity,
  const Device float*                 particlesLambda,
  const Device uint*                  gridCellParticleOffsets,
  const Device uint*                  gridCellParticleIndices,
  const Device uint*                  gridParticleCellIndex,
  const Device ParticleSharedData*    particleSharedData,
  Const XAB*                          systemBoundingBox,
  Const float*                        invRadius,
  constantKernelInput(int,            gridSize),
  constantKernelInput(int,            gridSizeExp),
  constantKernelInput(uint,           nodeCount)
  KERNEL_GLOBAL_ARGUMENTS
  KERNEL_THREAD_ARGUMENTS
  KERNEL_THREADGROUP_ARGUMENTS)
{
  uint particleIndex = threadIndex();

  if (particleIndex >= nodeCount)
  {
    return;
  }

  particleIndex = gridCellParticleIndices[particleIndex];

  const uint gridCellIndex = gridParticleCellIndex[particleIndex];

  float3 delta = constructFloat3(0.f);

  // current particle data
  ParticleStruct selfParticle = particlesPredictedOld[particleIndex];
  const IdentityInfo identity = selfParticle.identity;
  const ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(identity);
  const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];

  const float lambda = particlesLambda[particleIndex];

  const short3 particleGridCellIndex = constructShort3(
    gridCellIndex & (gridSize - 1),
    (gridCellIndex >> gridSizeExp) & (gridSize - 1),
    gridCellIndex >> (gridSizeExp << 1)
  );

#ifdef GRID_SOLVER_HASH_FUNCTION
  const float3 particleCellPosition = (selfParticle.position - systemBoundingBox->min) * invRadius[0];
#endif

  GRID_SOLVER_NEIGHBOUR_LOOP_BEGIN
    const uint2 indexRange = getRangeFromOffset(gridCellParticleOffsets, gridCellIndex);

    // loop through neighboring cells
    for (int otherParticlePointerIndex = indexRange.x; otherParticlePointerIndex < indexRange.y; otherParticlePointerIndex++)
    {
      const int otherNodeIndex = gridCellParticleIndices[otherParticlePointerIndex];
      const ParticleStruct otherParticle = particlesPredictedOld[otherNodeIndex];
      const float3 collisionVector = selfParticle.position - otherParticle.position;
      const float actualDistance = length(collisionVector);

      if (actualDistance >= sharedData.fluidKernelRadius || actualDistance <= COMPUTE_EPSILON || otherNodeIndex == particleIndex)
      {
        continue;
      }

      const float scorr = scorrFunction(actualDistance, sharedData.fluidKernelRadius);
      delta += collisionVector * ((lambda + particlesLambda[otherNodeIndex] + scorr) * spikyFunctionGradientVariable(actualDistance, sharedData.fluidKernelRadius));
  }
  GRID_SOLVER_NEIGHBOUR_LOOP_END

  selfParticle.position += delta * sharedData.invRestDensity * sharedData.fluidKernelFunctionConstant[1];
  selfParticle.identity = identity;

  particlesPredictedOld[particleIndex] = selfParticle;
  particlesNew[threadIndex()] = particlesOld[particleIndex];
}

/*
@kernel Calculate vorticity omega value.
@param particlesOmega Omega value for particles.
@param particlesDiff Particle velocity.
@param particlesPredictedOld Integrated particle position.
@param gridCellParticleOffsets Starting offset for each grid cell.
@param gridCellParticleIndices Output array for particle indices.
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
  const Device uint*                  gridCellParticleIndices,
  const Device uint*                  gridParticleCellIndex,
  const Device ParticleSharedData*    particleSharedData,
  Const XAB*                          systemBoundingBox,
  Const float*                        invRadius,
  constantKernelInput(int,            gridSize),
  constantKernelInput(int,            gridSizeExp),
  constantKernelInput(uint,           nodeCount)
  KERNEL_GLOBAL_ARGUMENTS
  KERNEL_THREAD_ARGUMENTS
  KERNEL_THREADGROUP_ARGUMENTS)
{
  uint particleIndex = threadIndex();

  if (particleIndex >= nodeCount)
  {
    return;
  }

  particleIndex = gridCellParticleIndices[particleIndex];

  const uint gridCellIndex = gridParticleCellIndex[particleIndex];

  float3 sumOmega = constructFloat3(0.f);

  // current particle data
  const ParticleStruct selfParticle = particlesPredictedOld[particleIndex];
  ParticleDifferential selfParticleDiff = particlesDiff[particleIndex];
  const IdentityInfo identity = selfParticle.identity;
  const ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(identity);
  const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];

  const short3 particleGridCellIndex = constructShort3(
    gridCellIndex & (gridSize - 1),
    (gridCellIndex >> gridSizeExp) & (gridSize - 1),
    gridCellIndex >> (gridSizeExp << 1)
  );

#ifdef GRID_SOLVER_HASH_FUNCTION
  const float3 particleCellPosition = (selfParticle.position - systemBoundingBox->min) * invRadius[0];
#endif

  GRID_SOLVER_NEIGHBOUR_LOOP_BEGIN
    const uint2 indexRange = getRangeFromOffset(gridCellParticleOffsets, gridCellIndex);

    // loop through neighboring cells
    for (int otherParticlePointerIndex = indexRange.x; otherParticlePointerIndex < indexRange.y; otherParticlePointerIndex++)
    {
      const int otherNodeIndex = gridCellParticleIndices[otherParticlePointerIndex];
      const ParticleStruct otherParticle = particlesPredictedOld[otherNodeIndex];
      const float3 collisionVector = selfParticle.position - otherParticle.position;
      const float actualDistance = length(collisionVector);

      if (actualDistance >= sharedData.fluidKernelRadius || actualDistance <= COMPUTE_EPSILON || otherNodeIndex == particleIndex)
      {
        continue;
      }

      const float3 velocityDiff = particlesDiff[otherNodeIndex].velocity - selfParticleDiff.velocity;
      sumOmega += cross(velocityDiff, collisionVector * (spikyFunctionGradientVariable(actualDistance, sharedData.fluidKernelRadius)));
  }
  GRID_SOLVER_NEIGHBOUR_LOOP_END

  particlesOmega[particleIndex] = sumOmega * sharedData.fluidKernelFunctionConstant[1];
}

/*
@kernel Calculate vorticity confinement and XSPH viscosity for particles.
@param particlesPredictedNew Reordered particle position.
@param particlesPredictedOld Integrated particle position.
@param particlesDiffNew Reordered particle velocity.
@param particlesDiffOld Particle velocity.
@param particlesOmega Omega value for particles.
@param gridCellParticleOffsets Starting offset for each grid cell.
@param gridCellParticleIndices Output array for particle indices.
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
  Device ParticleStruct*              particlesPredictedNew,
  const Device ParticleStruct*        particlesPredictedOld,
  Device ParticleDifferential*        particlesDiffNew,
  const Device ParticleDifferential*  particlesDiffOld,
  const Device float3*                particlesOmega,
  const Device uint*                  gridCellParticleOffsets,
  const Device uint*                  gridCellParticleIndices,
  const Device uint*                  gridParticleCellIndex,
  const Device ParticleSharedData*    particleSharedData,
  Const XAB*                          systemBoundingBox,
  Const float*                        invRadius,
  constantKernelInput(int,            gridSize),
  constantKernelInput(int,            gridSizeExp),
  constantKernelInput(uint,           nodeCount),
  constantKernelInput(float,          timeStep)
  KERNEL_GLOBAL_ARGUMENTS
  KERNEL_THREAD_ARGUMENTS
  KERNEL_THREADGROUP_ARGUMENTS)
{
  uint particleIndex = threadIndex();

  if (particleIndex >= nodeCount)
  {
    return;
  }

  particleIndex = gridCellParticleIndices[particleIndex];

  const uint gridCellIndex = gridParticleCellIndex[particleIndex];

  float3 delta = constructFloat3(0.f);
  float3 omegaDelta = constructFloat3(0.f);

  // current particle data
  ParticleStruct selfParticle = particlesPredictedOld[particleIndex];
  ParticleDifferential selfParticleDiff = particlesDiffOld[particleIndex];
  const IdentityInfo identity = selfParticle.identity;
  const ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(identity);
  const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];
  const float3 selfOmega = particlesOmega[particleIndex];

  const short3 particleGridCellIndex = constructShort3(
    gridCellIndex & (gridSize - 1),
    (gridCellIndex >> gridSizeExp) & (gridSize - 1),
    gridCellIndex >> (gridSizeExp << 1)
  );

#ifdef GRID_SOLVER_HASH_FUNCTION
  const float3 particleCellPosition = (selfParticle.position - systemBoundingBox->min) * invRadius[0];
#endif

  GRID_SOLVER_NEIGHBOUR_LOOP_BEGIN
    const uint2 indexRange = getRangeFromOffset(gridCellParticleOffsets, gridCellIndex);

    // loop through neighboring cells
    for (int otherParticlePointerIndex = indexRange.x; otherParticlePointerIndex < indexRange.y; otherParticlePointerIndex++)
    {
      const int otherNodeIndex = gridCellParticleIndices[otherParticlePointerIndex];
      const ParticleStruct otherParticle = particlesPredictedOld[otherNodeIndex];
      const float3 collisionVector = selfParticle.position - otherParticle.position;
      const float actualDistance = length(collisionVector);

      if (otherNodeIndex == particleIndex || actualDistance >= sharedData.fluidKernelRadius || actualDistance <= COMPUTE_EPSILON)
      {
        continue;
      }

      const float3 velocityDiff = particlesDiffOld[otherNodeIndex].velocity - selfParticleDiff.velocity;
      delta += velocityDiff * poly6FunctionVariable(actualDistance, sharedData.fluidKernelRadius);
      omegaDelta += collisionVector * (length(particlesOmega[otherNodeIndex]) * spikyFunctionGradientVariable(actualDistance, sharedData.fluidKernelRadius));
    }
  GRID_SOLVER_NEIGHBOUR_LOOP_END

  delta *= timeStep * sharedData.viscosity * sharedData.fluidKernelFunctionConstant[0];
  omegaDelta *= sharedData.fluidKernelFunctionConstant[1];

  const float omegaDeltaLength = length(omegaDelta);
  if (omegaDeltaLength > COMPUTE_EPSILON)
  {
    delta += cross(omegaDelta / omegaDeltaLength, selfOmega) * (0.01f * timeStep * timeStep * sharedData.sharedInvMass);
  }

  selfParticle.position += delta;
  selfParticle.identity = identity;
  particlesPredictedNew[threadIndex()] = selfParticle;
  particlesDiffNew[threadIndex()] = selfParticleDiff;
}

#endif
