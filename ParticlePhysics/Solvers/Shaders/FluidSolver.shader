#ifndef FLUID_SOLVER_SHADER
#define FLUID_SOLVER_SHADER

inline float pressureFunction(const float density, const Thread ParticleSharedData* sharedData)
{
  return sharedData->gasConstantK * (density - 1.f/sharedData->invRestDensity);
}

/*
 @kernel Resolve particle collisions.
 @param gridCellParticleOffsets Starting offset for each grid cell.
 @param gridCellParticleIndices Output array for particle indices.
 @param particlesPredictedNew Updated particle positions post collision processing.
 @param particlesPredictedOld Integrated particle position.
 @param particleSharedData Particle entity shared data.
 @param partitions Instance partition data.
 @param entityLocation Entity section data.
 @param gridParticleCellIndex Computed cell index for each particle.
 @param nodeCount Total nodes in the solver.
 @param occupiedCellCount Total active grid cells.
 */
Kernel void calculateDensity(
  Device float*                       particlesDensity,
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

  GRID_SOLVER_NEIGHBOUR_LOOP_BEGIN
    const uint2 indexRange = getRangeFromOffset(gridCellParticleOffsets, gridCellIndex);

    // batchwise iterate over indices in the cell
    for (int otherParticlePointerIndex = indexRange.x; otherParticlePointerIndex < indexRange.y; otherParticlePointerIndex++)
    {
      // iterate over each particle in the loaded batch
      const int otherNodeIndex = gridCellParticleIndices[otherParticlePointerIndex];
      const ParticleStruct otherParticle = particlesPredictedOld[otherNodeIndex];

      const float3 collisionVector = selfParticle.position - otherParticle.position;
      float actualDistance = length(collisionVector);

      density += select(0.f, poly6FunctionVariable(actualDistance, sharedData.fluidKernelRadius), actualDistance < sharedData.fluidKernelRadius);

      if (actualDistance <= COMPUTE_EPSILON || actualDistance >= sharedData.fluidKernelRadius || otherNodeIndex == particleIndex)
      {
        continue;
      }

      const float3 gradient = collisionVector * spikyFunctionGradientVariable(actualDistance, sharedData.fluidKernelRadius);
      sumGradientMagnitude += lengthSq(gradient);
      sumGradientVector += gradient;
    }
  GRID_SOLVER_NEIGHBOUR_LOOP_END

  sumGradientMagnitude += lengthSq(sumGradientVector);
  sumGradientMagnitude *= (sharedData.fluidKernelFunctionConstant[1] * sharedData.fluidKernelFunctionConstant[1]) / sharedData.sharedInvMass;

  density *= sharedData.fluidKernelFunctionConstant[0] / sharedData.sharedInvMass;

  particlesDensity[particleIndex] = density;
}

/*
 @kernel Resolve particle collisions.
 @param gridCellParticleOffsets Starting offset for each grid cell.
 @param gridCellParticleIndices Output array for particle indices.
 @param particlesPredictedNew Updated particle positions post collision processing.
 @param particlesPredictedOld Integrated particle position.
 @param particleSharedData Particle entity shared data.
 @param partitions Instance partition data.
 @param entityLocation Entity section data.
 @param gridParticleCellIndex Computed cell index for each particle.
 @param nodeCount Total nodes in the solver.
 @param occupiedCellCount Total active grid cells.
 */
Kernel void calculateForces(
  Device ParticleStruct*              particlesNew,
  const Device ParticleStruct*        particlesOld,
  Device ParticleStruct*              particlesPredictedNew,
  const Device float*                 particlesDensity,
  Device ParticleDifferential*        particleDiffNew,
  const Device ParticleDifferential*  particleDiff,
  const Device uint*                  gridCellParticleOffsets,
  const Device uint*                  gridCellParticleIndices,
  const Device uint*                  gridParticleCellIndex,
  const Device ParticleStruct*        particlesPredictedOld,
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

  float3 pressureForce = constructFloat3(0.f);
  float3 viscosityForce = constructFloat3(0.f);
  float3 colorGradient = constructFloat3(0.f);
  float3 colorLaplacian = constructFloat3(0.f);

  // current particle data
  ParticleStruct selfParticle = particlesPredictedOld[particleIndex];
  const float3 selfParticleDiff = particleDiff[particleIndex].velocity;
  const IdentityInfo identity = selfParticle.identity;
  const ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(identity);
  const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];
  const float density = particlesDensity[particleIndex];

  const short3 particleGridCellIndex = constructShort3(
    gridCellIndex & (gridSize - 1),
    (gridCellIndex >> gridSizeExp) & (gridSize - 1),
    gridCellIndex >> (gridSizeExp << 1)
  );

  const float selfPressureTerm = pressureFunction(density, &sharedData);

#ifdef GRID_SOLVER_HASH_FUNCTION
  const float3 particleCellPosition = (selfParticle.position - systemBoundingBox->min) * invRadius[0];
#endif

  GRID_SOLVER_NEIGHBOUR_LOOP_BEGIN
    const uint2 indexRange = getRangeFromOffset(gridCellParticleOffsets, gridCellIndex);

    // batchwise iterate over indices in the cell
    for (int otherParticlePointerIndex = indexRange.x; otherParticlePointerIndex < indexRange.y; otherParticlePointerIndex++)
    {
      // iterate over each particle in the loaded batch
      const int otherNodeIndex = gridCellParticleIndices[otherParticlePointerIndex];
      const ParticleStruct otherParticle = particlesPredictedOld[otherNodeIndex];

      const float3 collisionVector = selfParticle.position - otherParticle.position;
      float actualDistance = length(collisionVector);

      if (actualDistance <= COMPUTE_EPSILON || actualDistance >= sharedData.fluidKernelRadius || otherNodeIndex == particleIndex)
      {
        continue;
      }

      const float currentParticleDensity = particlesDensity[otherNodeIndex];
      const float currentParticleDensityInv = 1.f/currentParticleDensity;

      // force due to viscosity
      const float3 velocityVector = particleDiff[otherNodeIndex].velocity - selfParticleDiff;
      float viscosityTerm = sharedData.viscosity * viscosityFunctionLaplacianVariable(actualDistance, sharedData.fluidKernelRadius);

      viscosityForce += velocityVector * (viscosityTerm * currentParticleDensityInv);

      // force due to pressure
      const float pressureTerm = (selfPressureTerm + pressureFunction(currentParticleDensity, &sharedData)) * currentParticleDensityInv * 0.5f;

      pressureForce -= collisionVector * (pressureTerm * spikyFunctionGradientVariable(actualDistance, sharedData.fluidKernelRadius));

      colorGradient += collisionVector * poly6FunctionGradientVariable(actualDistance, sharedData.fluidKernelRadius) * currentParticleDensityInv;
      colorLaplacian += collisionVector * poly6FunctionLaplacianVariable(actualDistance, sharedData.fluidKernelRadius) * currentParticleDensityInv;
    }
  GRID_SOLVER_NEIGHBOUR_LOOP_END

  colorGradient *= sharedData.fluidKernelFunctionConstant[0];
  colorLaplacian *= sharedData.fluidKernelFunctionConstant[0];
  pressureForce *= sharedData.fluidKernelFunctionConstant[1];
  viscosityForce *= sharedData.fluidKernelFunctionConstant[2];

  float3 color = colorLaplacian;
  float colorLength = length(colorGradient);
  if (colorLength >= COMPUTE_EPSILON)
  {
    color *= -0.1f * colorGradient / colorLength;
  }

  selfParticle.position += (pressureForce + viscosityForce + color) * sqr(timeStep);
  selfParticle.identity = identity;

  particlesPredictedNew[threadIndex()] = selfParticle;
  particlesNew[threadIndex()] = particlesOld[particleIndex];
  particleDiffNew[threadIndex()] = particleDiff[particleIndex];
}

#endif
