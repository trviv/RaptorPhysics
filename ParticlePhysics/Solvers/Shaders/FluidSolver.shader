#ifndef FLUID_SOLVER_SHADER
#define FLUID_SOLVER_SHADER

inline float poly6Function(const float r, const float h)
{
#ifdef FLUID_USE_LAMBDA
  const float q = 2.0f * fabs(r) / h;
  if (q > 2.0f || q < COMPUTE_EPSILON)
  {
    return 0.0f;
  }
  else
  {
    const float a = 0.25f / (M_PI_F * h * h * h);
    return a * ((q > 1.0f) ? (2.0f - q) * (2.0f - q) * (2.0f - q) : ((3.0f * q - 6.0f) * q * q + 4.0f));
  }
#else
  const float x = (h * h - r * r) / (h * h * h);
  return (315.f / (64.f * M_PI_F)) * x * x * x;
#endif
}

inline float pressureFunction(const float density, const Thread ParticleSharedData* sharedData)
{
  return sharedData->gasConstantK * (density - 1.f/sharedData->invRestDensity);
}

inline float spikyFunction(const float r, const float h)
{
#ifdef FLUID_USE_LAMBDA
  const float q = 2.0f * r / h;
  if (q > 2.0f)
  {
    return 0.0f;
  }
  else
  {
    const float a = r / (M_PI_F * (q + COMPUTE_EPSILON) * h * h * h * h * h);
    return a * ((q > 1.0f) ? ((12.0f - 3.0f * q) * q - 12.0f) : ((9.0f * q - 12.0f) * q));
  }
#else
  const float x = (h - r) / (h * h);
  return (15.f / M_PI_F) * x * x * x;
#endif
}

inline float viscosityFunction(const float r, const float h)
{
#ifdef FLUID_USE_LAMBDA
  return poly6Function(r, h);
#else
//  float x = 1.f/h;
//  x *= x;
//  return (45.f / M_PI_F) * (h - r) * x * x * x;//(15.f / (2.f* M_PI_F * h * h * h)) * (-(r * r * r)/(2 * h * h * h)  + (r * r)/(h * h) + h/(2 * r) - 1);
  //return (15.f / (2.f * M_PI_F * h * h * h)) * (-(r * r * r)/(2 * h * h * h)  + (r * r)/(h * h) + h/(2 * r) - 1);
  const float t1 = (h * h * h);
  const float t2 = (r * r);
  return ((15.f / (2.f * M_PI_F)) / t1) * (-(r * t2)/(2.f * t1) + t2/(h * h) + h/(2.f * r) - 1);
#endif
}

inline float scorrFunction(const float r, const float h)
{
  const float corrK = 0.1f;
  const float corrDelQ = 0.1f;
  const int corrN = 4;
  const float x = poly6Function(r, h) / poly6Function(corrDelQ * h, h);
  //return -corrK * pow(x, corrN);
  const float t = x * x;
  return -corrK * t * t;
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
    const uint2 indexRange = getRangeFromOffset(gridCellParticleOffsets, gridCellIndex);

    // batchwise iterate over indices in the cell
    for (int otherParticlePointerIndex = indexRange.x; otherParticlePointerIndex < indexRange.y; otherParticlePointerIndex++)
    {
      // iterate over each particle in the loaded batch
      const int otherNodeIndex = gridCellParticleIndices[otherParticlePointerIndex];
      const ParticleStruct otherParticle = particlesPredictedOld[otherNodeIndex];

      const float3 collisionVector = selfParticle.position - otherParticle.position;
      float actualDistance = length(collisionVector);

      density += select(0.f, poly6Function(actualDistance, sharedData.fluidKernelRadius), actualDistance < sharedData.fluidKernelRadius);

      if (otherNodeIndex == particleIndex || actualDistance >= sharedData.fluidKernelRadius)
        continue;

      actualDistance = select(actualDistance, COMPUTE_EPSILON, actualDistance <= COMPUTE_EPSILON);

      const float3 gradient = collisionVector * spikyFunction(actualDistance, sharedData.fluidKernelRadius) / actualDistance;
      sumGradientMagnitude += lengthSq(gradient);
      sumGradientVector += gradient;
    }
  GRID_SOLVER_NEIGHBOUR_LOOP_END

  sumGradientMagnitude += lengthSq(sumGradientVector);
  sumGradientMagnitude /= sharedData.sharedInvMass;

  density /= sharedData.sharedInvMass;

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

#ifdef MARK_COLLIDED_PARTICLES
  bool collided = false;
#endif

  float force = 0.f;
  float3 delta = constructFloat3(0.f);

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

      if (otherNodeIndex == particleIndex || actualDistance >= sharedData.fluidKernelRadius)
      {
        continue;
      }

      actualDistance = select(actualDistance, COMPUTE_EPSILON, actualDistance <= COMPUTE_EPSILON);

      const float3 velocityVector = particleDiff[otherNodeIndex].velocity - selfParticleDiff;
      float viscosityTerm = sharedData.viscosity * viscosityFunction(actualDistance, sharedData.fluidKernelRadius);
      const float currentParticleDensity = particlesDensity[otherNodeIndex];
      const float currentParticleDensityInv = 1.f/currentParticleDensity;
      const float pressureTerm = (pressureFunction(density, &sharedData) + pressureFunction(currentParticleDensity, &sharedData)) * currentParticleDensityInv * 0.5f;
      const float distanceFunction = spikyFunction(actualDistance, sharedData.fluidKernelRadius);

      // force due to pressure
      delta -= collisionVector * (sqr(timeStep) * pressureTerm * distanceFunction / actualDistance);

      // force due to viscosity
      delta += velocityVector * (timeStep * viscosityTerm * currentParticleDensityInv);
    }
  GRID_SOLVER_NEIGHBOUR_LOOP_END

  delta *= sharedData.invRestDensity;
  selfParticle.position += delta * sharedData.sharedInvMass;
  selfParticle.identity = identity;

  particlesPredictedNew[threadIndex()] = selfParticle;
  particlesNew[threadIndex()] = particlesOld[particleIndex];
  particleDiffNew[threadIndex()] = particleDiff[particleIndex];
}

#endif
