#ifndef FLUID_SOLVER_SHADER
#define FLUID_SOLVER_SHADER

inline float pressureFunction(const float density, const Thread ParticleSharedData* sharedData)
{
  return sharedData->gasConstantK * (density - 1.f/sharedData->invRestDensity);
}

/*
@kernel Resolve particle collisions.
@param particlesDensity Particles density.
@param gridCellParticleOffsets Starting offset for each grid cell.
@param particlesPredicted Integrated particle position.
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

  // current particle data
  float density = 0.f;
  float sumGradientMagnitude = 0.f;
  float3 sumGradientVector = constructFloat3(0.f);

  const ParticleStruct selfParticle = particlesPredicted[particleIndex];
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
    for (int otherNodeIndex = indexRange.x; otherNodeIndex < indexRange.y; otherNodeIndex++)
    {
      // iterate over each particle in the loaded batch
      const ParticleStruct otherParticle = particlesPredicted[otherNodeIndex];

      const float3 collisionVector = selfParticle.position - otherParticle.position;
      float actualDistance = length(collisionVector);

      density += select(0.f, poly6FunctionVariable(actualDistance, sharedData.fluidSolverData.fluidKernelRadius), actualDistance < sharedData.fluidSolverData.fluidKernelRadius);

      if (actualDistance <= COMPUTE_EPSILON || actualDistance >= sharedData.fluidSolverData.fluidKernelRadius || otherNodeIndex == particleIndex)
      {
        continue;
      }

      const float3 gradient = collisionVector * spikyFunctionGradientVariable(actualDistance, sharedData.fluidSolverData.fluidKernelRadius);
      sumGradientMagnitude += lengthSq(gradient);
      sumGradientVector += gradient;
    }
  GRID_SOLVER_NEIGHBOUR_LOOP_END

  sumGradientMagnitude += lengthSq(sumGradientVector);
  sumGradientMagnitude *= (sharedData.fluidSolverData.fluidKernelFunctionConstant[1] * sharedData.fluidSolverData.fluidKernelFunctionConstant[1]) / sharedData.sharedInvMass;

  density *= sharedData.fluidSolverData.fluidKernelFunctionConstant[0] / sharedData.sharedInvMass;

  particlesDensity[particleIndex] = density;
}

/*
@kernel Resolve particle collisions.
@param gridCellParticleOffsets Starting offset for each grid cell.
@param particlesPredictedNew Updated particle positions post collision processing.
@param particlesDensity Particles density.
@param particlesPredictedOld Integrated particle position.
@param particleSharedData Particle entity shared data.
@param partitions Instance partition data.
@param entityLocation Entity section data.
@param gridParticleCellIndex Computed cell index for each particle.
@param nodeCount Total nodes in the solver.
@param occupiedCellCount Total active grid cells.
*/
Kernel void calculateForces(
  Device ParticleStruct*              particlesPredictedNew,
  const Device float*                 particlesDensity,
  const Device ParticleDifferential*  particleDiff,
  const Device uint*                  gridCellParticleOffsets,
  const Device uint*                  gridParticleCellIndex,
  const Device ParticleStruct*        particlesPredictedOld,
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
    for (int otherNodeIndex = indexRange.x; otherNodeIndex < indexRange.y; otherNodeIndex++)
    {
      // iterate over each particle in the loaded batch
      const ParticleStruct otherParticle = particlesPredictedOld[otherNodeIndex];

      const float3 collisionVector = selfParticle.position - otherParticle.position;
      float actualDistance = length(collisionVector);

      if (actualDistance <= COMPUTE_EPSILON || actualDistance >= sharedData.fluidSolverData.fluidKernelRadius || otherNodeIndex == particleIndex)
      {
        continue;
      }

      const float currentParticleDensity = particlesDensity[otherNodeIndex];
      const float currentParticleDensityInv = 1.f/currentParticleDensity;

      // force due to viscosity
      const float3 velocityVector = particleDiff[otherNodeIndex].velocity - selfParticleDiff;
      float viscosityTerm = sharedData.viscosity * viscosityFunctionLaplacianVariable(actualDistance, sharedData.fluidSolverData.fluidKernelRadius);

      viscosityForce += velocityVector * (viscosityTerm * currentParticleDensityInv);

      // force due to pressure
      const float pressureTerm = (selfPressureTerm + pressureFunction(currentParticleDensity, &sharedData)) * currentParticleDensityInv * 0.5f;

      pressureForce -= collisionVector * (pressureTerm * spikyFunctionGradientVariable(actualDistance, sharedData.fluidSolverData.fluidKernelRadius));

      colorGradient += collisionVector * poly6FunctionGradientVariable(actualDistance, sharedData.fluidSolverData.fluidKernelRadius) * currentParticleDensityInv;
      colorLaplacian += collisionVector * poly6FunctionLaplacianVariable(actualDistance, sharedData.fluidSolverData.fluidKernelRadius) * currentParticleDensityInv;
    }
  GRID_SOLVER_NEIGHBOUR_LOOP_END

  colorGradient *= sharedData.fluidSolverData.fluidKernelFunctionConstant[0];
  colorLaplacian *= sharedData.fluidSolverData.fluidKernelFunctionConstant[0];
  pressureForce *= sharedData.fluidSolverData.fluidKernelFunctionConstant[1];
  viscosityForce *= sharedData.fluidSolverData.fluidKernelFunctionConstant[2];

  float3 color = colorLaplacian;
  float colorLength = length(colorGradient);
  if (colorLength >= COMPUTE_EPSILON)
  {
    color *= -0.1f * colorGradient / colorLength;
  }

  selfParticle.position += (pressureForce + viscosityForce + color) * sqr(timeStep);
  selfParticle.identity = identity;

  particlesPredictedNew[particleIndex] = selfParticle;
}

#endif
