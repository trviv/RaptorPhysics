#ifndef FLUID_SOLVER_SHADER
#define FLUID_SOLVER_SHADER

#define successiveOverRealaxation 1.5f
#define FLUID_SIM_EPSILON         1.f

inline float poly6Function(const float r, const float h)
{
  const float x = (h * h - r * r) / (h * h * h);
  return (315.f / (64.f * M_PI_F)) * x * x * x;
}

inline float pressureFunction(const float density, const Thread ParticleSharedData* sharedData)
{
  return sharedData->gasConstantK * (density - 1.f/sharedData->invRestDensity);
//  return density;
}

inline float spikyFunction(const float r, const float h)
{
  const float x = (h - r) / (h * h);
  return (15.f / M_PI_F) * x * x * x;
}

inline float viscosityFunction(const float r, const float h)
{
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
  const Device ParticleStruct*        particlesPredictedOld,
  const Device ParticleSharedData*    particleSharedData,
  constantKernelInput(int,            gridSize)
  KERNEL_GLOBAL_ARGUMENTS
  KERNEL_THREAD_ARGUMENTS
  KERNEL_THREADGROUP_ARGUMENTS)
{
  const uint gridCellIndex = gridCompactCellIndices[threadGroupIndex()];

  // particle index buffer
  const int indexBufferEnd = gridCellParticleOffsets[gridCellIndex];
  const int indexBufferCount = gridCellIndexCount[gridCellIndex];

  // batchwise iterate over indices in the cell
  for (int particlePointerIndex = indexBufferEnd - indexBufferCount + threadLocalIndex(); particlePointerIndex < indexBufferEnd; particlePointerIndex += threadGroupSize())
  {
#ifdef MARK_COLLIDED_PARTICLES
    bool collided = false;
#endif

    // current particle data
    float density = 0.f;
    float gradientMagnitude = 0.f;
    float3 accumulatedGradient = constructFloat3(0.f);

    const int particleIndex = gridCellParticleIndices[particlePointerIndex];
    const ParticleStruct currentParticle = particlesPredictedOld[particleIndex];
    const IdentityInfo identity = currentParticle.identity;
    const ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(identity);
    const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];

    for (short k=-1; k<2; k++)
    {
      const int z = (gridCellIndex / (gridSize * gridSize) + k + gridSize) & (gridSize - 1);
      for (short j=-1; j<2; j++)
      {
        const int y = ((gridCellIndex / gridSize) + j + gridSize) & (gridSize - 1);
        for (short i=-1; i<2; i++)
        {
          const int x = (gridCellIndex + i + gridSize) & (gridSize - 1);
          const int gridCellIndex2 = x + gridSize * (y + z * gridSize);
          int indexBufferCount2 = gridCellIndexCount[gridCellIndex2];

          if (indexBufferCount2 == 0)
          {
            continue;
          }

          const int indexBufferEnd2 = gridCellParticleOffsets[gridCellIndex2];

          // batchwise iterate over indices in the cell
          for (int otherParticlePointerIndex = indexBufferEnd2 - indexBufferCount2; otherParticlePointerIndex < indexBufferEnd2; otherParticlePointerIndex++)
          {
            // iterate over each particle in the loaded batch
            const int currentNodeIndex = gridCellParticleIndices[otherParticlePointerIndex];
            const ParticleStruct otherParticle = particlesPredictedOld[currentNodeIndex];

            const float3 collisionVector = currentParticle.position - otherParticle.position;
            const float actualDistance = length(collisionVector);

            density += select(0.f, poly6Function(actualDistance, sharedData.fluidKernelRadius), actualDistance < sharedData.fluidKernelRadius);

            if (currentNodeIndex == particleIndex)
            {
              continue;
            }

            float3 gradient = collisionVector * select(0.f, spikyFunction(actualDistance, sharedData.fluidKernelRadius), actualDistance < sharedData.fluidKernelRadius) / actualDistance;
//            float3 gradient = constructFloat3(0.f);
//            if (actualDistance < sharedData.fluidKernelRadius)
//            {
//              gradient = - collisionVector * spikyFunction(actualDistance, sharedData.fluidKernelRadius) / actualDistance;
//            }
            gradientMagnitude += dot(gradient, gradient);
            accumulatedGradient += gradient;
          }
        }
      }
    }

    gradientMagnitude += dot(accumulatedGradient, accumulatedGradient);
    gradientMagnitude *= sharedData.invRestDensity;

//    density = (density * sharedData.invRestDensity / sharedData.sharedInvMass - 1.f);
    density *= 1.f/sharedData.sharedInvMass;
//    density = max(sharedData.invRestDensity, density);

    particlesDensity[particleIndex] = density;
//    particlesLambda[particleIndex] = - density  / (gradientMagnitude + FLUID_SIM_EPSILON);
  }
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
  Device ParticleStruct*              particlesPredictedNew,
  const Device float*                 particlesDensity,
  const Device float*                 particlesLambda,
  const Device ParticleDifferential*  particleDiff,
  const Device uint*                  gridCompactCellIndices,
  const Device uint*                  gridCellParticleOffsets,
  const Device uint*                  gridCellIndexCount,
  const Device uint*                  gridCellParticleIndices,
  const Device ParticleStruct*        particlesPredictedOld,
  const Device ParticleSharedData*    particleSharedData,
  constantKernelInput(int,            gridSize)
  KERNEL_GLOBAL_ARGUMENTS
  KERNEL_THREAD_ARGUMENTS
  KERNEL_THREADGROUP_ARGUMENTS)
{
  const uint gridCellIndex = gridCompactCellIndices[threadGroupIndex()];

  // particle index buffer
  const int indexBufferEnd = gridCellParticleOffsets[gridCellIndex];
  const int indexBufferCount = gridCellIndexCount[gridCellIndex];

  // batchwise iterate over indices in the cell
  for (int particlePointerIndex = indexBufferEnd - indexBufferCount + threadLocalIndex(); particlePointerIndex < indexBufferEnd; particlePointerIndex += threadGroupSize())
  {
#ifdef MARK_COLLIDED_PARTICLES
    bool collided = false;
#endif

    float force = 0.f;
    float3 delta = constructFloat3(0.f);

    // current particle data
    const int particleIndex = gridCellParticleIndices[particlePointerIndex];
    ParticleStruct currentParticle = particlesPredictedOld[particleIndex];
    const IdentityInfo identity = currentParticle.identity;
    const ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(identity);
    const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];
    const float mass = 1.f/sharedData.sharedInvMass;
    const float lambda = particlesLambda[particleIndex];
    const float density = particlesDensity[particleIndex];

    for (short k=-1; k<2; k++)
    {
      const int z = (gridCellIndex / (gridSize * gridSize) + k + gridSize) & (gridSize - 1);
      for (short j=-1; j<2; j++)
      {
        const int y = ((gridCellIndex / gridSize) + j + gridSize) & (gridSize - 1);
        for (short i=-1; i<2; i++)
        {
          const int x = (gridCellIndex + i + gridSize) & (gridSize - 1);
          const int gridCellIndex2 = x + gridSize * (y + z * gridSize);
          int indexBufferCount2 = gridCellIndexCount[gridCellIndex2];

          if (indexBufferCount2 == 0)
          {
            continue;
          }

          const int indexBufferEnd2 = gridCellParticleOffsets[gridCellIndex2];

          // batchwise iterate over indices in the cell
          for (int otherParticlePointerIndex = indexBufferEnd2 - indexBufferCount2; otherParticlePointerIndex < indexBufferEnd2; otherParticlePointerIndex++)
          {
            // iterate over each particle in the loaded batch
            const int currentNodeIndex = gridCellParticleIndices[otherParticlePointerIndex];
            const ParticleStruct otherParticle = particlesPredictedOld[currentNodeIndex];

            const float3 collisionVector = currentParticle.position - otherParticle.position;
            const float actualDistance = length(collisionVector);

//              // force due to pressure
//              force += -mass * (pressure + pressureFunction(particleDensity, sharedData)) / (2.f * particleDensity) *
//                select(0.f, select(0.f, spikyFunction(actualDistance, kernelSize), actualDistance < kernelSize), actualDistance > 0.f);

            if (currentNodeIndex == particleIndex) continue;

//            delta += sharedData.invRestDensity * collisionVector * ((lambda + particlesLambda[currentNodeIndex] + scorrFunction(actualDistance, sharedData.fluidKernelRadius)) *
//                select(0.f, spikyFunction(actualDistance, sharedData.fluidKernelRadius), actualDistance < sharedData.fluidKernelRadius) / actualDistance);
            const float pressureTerm = (pressureFunction(density, &sharedData) + pressureFunction(particlesDensity[currentNodeIndex], &sharedData))/(2*particlesDensity[currentNodeIndex]);
            const float distanceFunction = select(0.f, spikyFunction(actualDistance, sharedData.fluidKernelRadius), actualDistance < sharedData.fluidKernelRadius);
            delta -= collisionVector * (sqr(1.f/60.f) * pressureTerm * distanceFunction / actualDistance);

            float3 velocityVector = particleDiff[currentNodeIndex].velocity - particleDiff[particleIndex].velocity;
            float viscosityTerm = select(0.f, viscosityFunction(actualDistance, sharedData.fluidKernelRadius), actualDistance < sharedData.fluidKernelRadius);
            delta += velocityVector * ((1.f/60.f) * viscosityTerm / particlesDensity[currentNodeIndex]);
          }
        }
      }
    }

    currentParticle.position += delta * 1.f/sharedData.sharedInvMass;
    currentParticle.identity = identity;
    particlesPredictedNew[particleIndex] = currentParticle;
  }
}

#endif
