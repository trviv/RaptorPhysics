#ifndef FLUID_SOLVER_SHADER
#define FLUID_SOLVER_SHADER

#define successiveOverRealaxation 1.5f
#define FLUID_SIM_EPSILON         1.f
//#define FLUID_USE_LAMBDA

inline float poly6Function(const float r, const float h)
{
  const float x = (h * h - r * r) / (h * h * h);
  return (315.f / (64.f * M_PI_F)) * x * x * x;
}

inline float pressureFunction(const float density, const Thread ParticleSharedData* sharedData)
{
  return sharedData->gasConstantK * (density - 1.f/sharedData->invRestDensity);
}

inline float spikyFunction(const float r, const float h)
{
  const float x = (h - r) / (h * h);
  return (15.f / M_PI_F) * x * x * x;
}

inline float viscosityFunction(const float r, const float h)
{
//  float x = 1.f/h;
//  x *= x;
//  return (45.f / M_PI_F) * (h - r) * x * x * x;//(15.f / (2.f* M_PI_F * h * h * h)) * (-(r * r * r)/(2 * h * h * h)  + (r * r)/(h * h) + h/(2 * r) - 1);
  return (15.f / (2.f* M_PI_F * h * h * h)) * (-(r * r * r)/(2 * h * h * h)  + (r * r)/(h * h) + h/(2 * r) - 1);
}

inline float scorrFunction(const float r, const float h)
{
  const float corrK = 0.1f;
  const float corrDelQ = 0.1f;
  const int corrN = 4;
  const float x = poly6Function(r, h) / poly6Function(corrDelQ * h, h);
  return -corrK * pow(x, corrN);
}

/*
 @kernel Resolve particle collisions.
 @param gridCompactCellIndices Map to the cell index to be processed.
 @param gridCellParticleOffsets Starting offset for each grid cell.
 @param gridCellIndexCount Particle count for each grid cell.
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
  Device float*                       particlesLambda,
  const Device uint*                  gridCompactCellIndices,
  const Device uint*                  gridCellParticleOffsets,
  const Device uint*                  gridCellIndexCount,
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

#ifdef MARK_COLLIDED_PARTICLES
  bool collided = false;
#endif

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
    int indexBufferCount = gridCellIndexCount[gridCellIndex];

    if (indexBufferCount == 0)
    {
      continue;
    }

    const int indexBufferEnd = gridCellParticleOffsets[gridCellIndex];

    // batchwise iterate over indices in the cell
    for (int otherParticlePointerIndex = indexBufferEnd - indexBufferCount; otherParticlePointerIndex < indexBufferEnd; otherParticlePointerIndex++)
    {
      // iterate over each particle in the loaded batch
      const int otherNodeIndex = gridCellParticleIndices[otherParticlePointerIndex];
      const ParticleStruct otherParticle = particlesPredictedOld[otherNodeIndex];

      const float3 collisionVector = selfParticle.position - otherParticle.position;
      float actualDistance = length(collisionVector);

      density += select(0.f, poly6Function(actualDistance, sharedData.fluidKernelRadius), actualDistance < sharedData.fluidKernelRadius);

      if (otherNodeIndex == particleIndex)
      {
        continue;
      }

      actualDistance = select(actualDistance, COMPUTE_EPSILON, actualDistance <= COMPUTE_EPSILON);

      float3 gradient = collisionVector * select(0.f, spikyFunction(actualDistance, sharedData.fluidKernelRadius), actualDistance < sharedData.fluidKernelRadius) / actualDistance;
      sumGradientMagnitude += dot(gradient, gradient);
      sumGradientVector += gradient;
    }
  GRID_SOLVER_NEIGHBOUR_LOOP_END

  sumGradientMagnitude += dot(sumGradientVector, sumGradientVector);
  //sumGradientMagnitude *= sharedData.invRestDensity;

  density /= sharedData.sharedInvMass;

#ifdef FLUID_USE_LAMBDA
  density = density * sharedData.invRestDensity - 1.f;
  particlesDensity[particleIndex] = -density / (sumGradientMagnitude + 10.1f) ;
#else
  particlesDensity[particleIndex] = density;
#endif
}

/*
 @kernel Resolve particle collisions.
 @param gridCompactCellIndices Map to the cell index to be processed.
 @param gridCellParticleOffsets Starting offset for each grid cell.
 @param gridCellIndexCount Particle count for each grid cell.
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
  const Device float*                 particlesLambda,
  Device ParticleDifferential*        particleDiffNew,
  const Device ParticleDifferential*  particleDiff,
  const Device uint*                  gridCompactCellIndices,
  const Device uint*                  gridCellParticleOffsets,
  const Device uint*                  gridCellIndexCount,
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

#ifdef MARK_COLLIDED_PARTICLES
  bool collided = false;
#endif

  float force = 0.f;
  float3 delta = constructFloat3(0.f);

  // current particle data
  ParticleStruct selfParticle = particlesPredictedOld[particleIndex];
  const IdentityInfo identity = selfParticle.identity;
  const ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(identity);
  const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];
  const float lambda = particlesLambda[particleIndex];
  const float density = particlesDensity[particleIndex];

  const short3 particleGridCellIndex = constructShort3(
    gridCellIndex & (gridSize - 1),
    (gridCellIndex >> gridSizeExp) & (gridSize - 1),
    gridCellIndex >> (gridSizeExp << 1)
  );

#ifdef GRID_SOLVER_HASH_FUNCTION
  const float3 particleCellPosition = (selfParticle.position - systemBoundingBox->min) * invRadius[0];
#endif

  GRID_SOLVER_NEIGHBOUR_LOOP_BEGIN
    int indexBufferCount = gridCellIndexCount[gridCellIndex];

    if (indexBufferCount == 0)
    {
      continue;
    }

    const int indexBufferEnd = gridCellParticleOffsets[gridCellIndex];

    // batchwise iterate over indices in the cell
    for (int otherParticlePointerIndex = indexBufferEnd - indexBufferCount; otherParticlePointerIndex < indexBufferEnd; otherParticlePointerIndex++)
    {
      // iterate over each particle in the loaded batch
      const int otherNodeIndex = gridCellParticleIndices[otherParticlePointerIndex];
      const ParticleStruct otherParticle = particlesPredictedOld[otherNodeIndex];

      const float3 collisionVector = selfParticle.position - otherParticle.position;
      float actualDistance = length(collisionVector);

      if (otherNodeIndex == particleIndex)
      {
        continue;
      }

      actualDistance = select(actualDistance, COMPUTE_EPSILON, actualDistance <= COMPUTE_EPSILON);

      const float currentParticleDensity = particlesDensity[otherNodeIndex];

//            delta += sharedData.invRestDensity * collisionVector * ((lambda + particlesLambda[otherNodeIndex] + scorrFunction(actualDistance, sharedData.fluidKernelRadius)) *
//                select(0.f, spikyFunction(actualDistance, sharedData.fluidKernelRadius), actualDistance < sharedData.fluidKernelRadius) / actualDistance);
      const float pressureTerm = (pressureFunction(density, &sharedData) + pressureFunction(currentParticleDensity, &sharedData)) / (2.f * currentParticleDensity);
      const float distanceFunction = select(0.f, spikyFunction(actualDistance, sharedData.fluidKernelRadius), actualDistance < sharedData.fluidKernelRadius);
      float timeStep = 1.f/60.f;

#ifndef FLUID_USE_LAMBDA
      // force due to pressure
      delta -= collisionVector * (sqr(timeStep) * pressureTerm * distanceFunction / actualDistance);

      float3 velocityVector = particleDiff[otherNodeIndex].velocity - particleDiff[particleIndex].velocity;
      float viscosityTerm = sharedData.viscosity * select(0.f, viscosityFunction(actualDistance, sharedData.fluidKernelRadius), actualDistance < sharedData.fluidKernelRadius);

      // force due to viscosity
      delta += velocityVector * (timeStep * viscosityTerm / currentParticleDensity);
#else
      const float corr = scorrFunction(actualDistance, sharedData.fluidKernelRadius);
      delta += collisionVector * ((density + currentParticleDensity + corr) *
        select(0.f, spikyFunction(actualDistance, sharedData.fluidKernelRadius), actualDistance < sharedData.fluidKernelRadius) / actualDistance);
#endif
    }
  GRID_SOLVER_NEIGHBOUR_LOOP_END

  delta *= sharedData.invRestDensity;
  selfParticle.position += delta / sharedData.sharedInvMass;
  selfParticle.identity = identity;

  particlesPredictedNew[threadIndex()] = selfParticle;
  particlesNew[threadIndex()] = particlesOld[particleIndex];
  particleDiffNew[threadIndex()] = particleDiff[particleIndex];
}

#endif
