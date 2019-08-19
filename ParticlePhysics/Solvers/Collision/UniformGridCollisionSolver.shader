#ifndef COLLISION_SOLVER_SHADER
#define COLLISION_SOLVER_SHADER

inline uint3 quantizePosition(const float3 position, const uint gridSize)
{
  const float3 particlePredictedScaled = fabs(position.xyz);
  return constructUint3(particlePredictedScaled.x, particlePredictedScaled.y, particlePredictedScaled.z) & constructUint3(gridSize - 1);
}

/*
@kernel Compute and store the cell index for each particle, and atomically increment the cell count for grid cell.
@param gridCellIndexCount Particle count for each grid cell.
@param gridParticleCellIndex Computed cell index for each particle.
@param particlesPredicted Integrated particle position.
@param nodeCount Total nodes in the solver.
@param gridSize Size of grid in one dimension.
*/
Kernel void createGridCellHistogram(
  atomicKernelInput(uint,           gridCellIndexCount),
  Device uint*                      gridParticleCellIndex,
  const Device ParticleStruct*      particlesPredicted,
  constantKernelInput(uint,         nodeCount),
  constantKernelInput(uint,         gridSize)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint index = threadIndex();

  if (index < nodeCount)
  {
    const uint3 quantizedPosition = quantizePosition(particlesPredicted[index].position, gridSize);
    const uint gridCountOffset = (quantizedPosition.z * gridSize + quantizedPosition.y) * gridSize + quantizedPosition.x;

    gridParticleCellIndex[index] = gridCountOffset;
    atomicAdd(&gridCellIndexCount[gridCountOffset], 1);
  }
}

/*
@kernel Store the particle indices for each grid cell in a continuous array.
@param gridCellParticleIndices Output array for particle indices.
@param gridCellParticleOffsets Starting offset for each grid cell.
@param gridParticleCellIndex Computed cell index for each particle.
@param nodeCount Total nodes in the solver.
*/
Kernel void createGridCellArrays(
  Device uint*                      gridCellParticleIndices,
  atomicKernelInput(uint,           gridCellParticleOffsets),
  const Device uint*                gridParticleCellIndex,
  constantKernelInput(uint,         nodeCount)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint index = threadIndex();

  if (index < nodeCount)
  {
    const uint gridCountOffset = gridParticleCellIndex[index];
    const uint offset = atomicAdd(&gridCellParticleOffsets[gridCountOffset], 1);

    gridCellParticleIndices[offset] = index;
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
@param particleCollisionData Array containing particle SDF mass and radius data.
@param particleSharedData Particle entity shared data.
@param partitions Instance partition data.
@param entityLocation Entity section data.
@param globalOffsets Offsets to particle nodes all the solvers.
@param gridParticleCellIndex Computed cell index for each particle.
@param nodeCount Total nodes in the solver.
@param occupiedCellCount Total active grid cells.
*/
Kernel void applyCollisions(
  const Device uint*                  gridCompactCellIndices,
  const Device uint*                  gridCellParticleOffsets,
  const Device uint*                  gridCellIndexCount,
  const Device uint*                  gridCellParticleIndices,
  Device ParticleStruct*              particlesPredictedNew,
  Device ParticleStruct*              particles2,
  const Device ParticleStruct*        particlesInit,
  const Device ParticleStruct*        particlesPredictedOld,
#ifdef MARK_COLLIDED_PARTICLES
  Device ParticleCollisionData*       particleCollisionData,
#else
  const Device ParticleCollisionData* particleCollisionData,
#endif
  const Device ParticleSharedData*    particleSharedData,
  Const PhySystemOffsets*             globalOffsets,
  constantKernelInput(int,            gridSize),
  constantKernelInput(uint,           stablizationPass)
  KERNEL_GLOBAL_ARGUMENTS
  KERNEL_THREAD_ARGUMENTS
  KERNEL_THREADGROUP_ARGUMENTS)
{
  const uint gridCellIndex = gridCompactCellIndices[threadGroupIndex()];

  // particle index buffer
  const int end = gridCellParticleOffsets[gridCellIndex];
  int count = gridCellIndexCount[gridCellIndex];

  const int batchBegin = end - count + threadLocalIndex();
  const int batchEnd = end;
  short collisionCount = 0;

  // batchwise iterate over indices in the cell
  for (int baseIndex = batchBegin; baseIndex < batchEnd; baseIndex += threadGroupSize())
  {
    // current particle data
    int particleIndex;
    float3 delta = constructFloat3(0.f);
    IdentityInfo identity;
    ParticleStruct currentParticle;
    ParticleStruct particleInit;
    ParticleSharedData sharedData;
    ParticleCollisionData collisionData;
    float sdfMagnitude;

#ifdef MARK_COLLIDED_PARTICLES
    bool collided = false;
#endif

    if (baseIndex < end)
    { // read this batch
      particleIndex = gridCellParticleIndices[baseIndex];
      currentParticle = particlesPredictedOld[particleIndex];
      particleInit = particlesInit[particleIndex];
      identity = currentParticle.identity;

      ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(identity);
      const PhySystemOffsets phySystemOffsets = globalOffsets[nodeIdentity.solverType];

      nodeIdentity.entityId += phySystemOffsets.globalSolverOffset;
      nodeIdentity.instanceId += phySystemOffsets.globalInstanceOffset;

      sharedData = particleSharedData[nodeIdentity.entityId];
      collisionData = particleCollisionData[particleIndex];

      sdfMagnitude = length(collisionData.transformedSdfGradient);

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
            int count2 = gridCellIndexCount[gridCellIndex2];

            if (count2 == 0)
            {
              continue;
            }

            const int end2 = gridCellParticleOffsets[gridCellIndex2];

            // batchwise iterate over indices in the cell
            for (int otherIndex = end2 - count2; otherIndex < end2; otherIndex++)
            {
              // iterate over each particle in the loaded batch
              const int currentNodeIndex = gridCellParticleIndices[otherIndex];

              const ParticleStruct otherParticle = particlesPredictedOld[currentNodeIndex];
              const ParticleStruct otherParticleInit = particlesInit[currentNodeIndex];
              delta += sharedData.collisionDamping * processParticleCollision(&currentParticle, &particleInit, &otherParticle, &otherParticleInit,
                &collisionData, &sharedData, currentNodeIndex, particleIndex, sdfMagnitude, &collisionCount,
#ifdef MARK_COLLIDED_PARTICLES
                particleCollisionData, &collided);
#else
                particleCollisionData);
#endif
            }
          }
        }
      }

      // apply boundary
      delta += boundaryCollision(&currentParticle, &collisionData);

//      if (collisionCount)
//      {
//        delta /= collisionCount;
//      }

      currentParticle.position += delta;
      currentParticle.identity = identity;

      particlesPredictedNew[particleIndex] = currentParticle;

      if (stablizationPass)
      {
        particles2[particleIndex].position += delta;
        particles2[particleIndex].identity = identity;
      }

#ifdef MARK_COLLIDED_PARTICLES
      particleCollisionData[particleIndex].radius = fabs(collisionData.radius) * (collided ? -1.f : 1.f);
#endif
    }
  }
}

#endif
