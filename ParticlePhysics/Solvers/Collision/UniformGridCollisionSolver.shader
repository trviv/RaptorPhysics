#ifndef GRID_SOLVER_SHADER
#define GRID_SOLVER_SHADER

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
  const Device ParticleStruct*      particles,
  Const XAB*                        systemBoundingBox,
  Const float*                      radius,
  constantKernelInput(uint,         nodeCount),
  constantKernelInput(uint,         gridSize)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint index = threadIndex();

  if (index < nodeCount)
  {
    const float3 inverseMergedBoxSize = ((float)gridSize) / max(gridSize * radius[0], systemBoundingBox->max - systemBoundingBox->min);
    const int3 quantizedPosition = convertInt3((particles[index].position - systemBoundingBox->min) * inverseMergedBoxSize);
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
@param systemSettings Settings for the physics system.
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
  Device ParticleStruct*              particlesInit,
  const Device ParticleDifferential*  particlesDiff,
  const Device ParticleStruct*        particlesPredictedOld,
#ifdef MARK_COLLIDED_PARTICLES
  Device ParticleCollisionData*       particleCollisionData,
#else
  const Device ParticleCollisionData* particleCollisionData,
#endif
  const Device ParticleSharedData*    particleSharedData,
  Const PhySystemSettings*            systemSettings,
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
    ParticleDifferential selfParticleDiff;
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
      selfParticleDiff = particlesDiff[particleIndex];
      identity = currentParticle.identity;

      ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(identity);
      const PhySystemOffsets phySystemOffsets = systemSettings->globalOffsets[nodeIdentity.solverType];

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
              const ParticleDifferential otherParticleDiff = particlesDiff[currentNodeIndex];
              delta += sharedData.collisionDamping * processParticleCollision(&currentParticle, &selfParticleDiff, &otherParticle, &otherParticleDiff,
                &collisionData, &sharedData, currentNodeIndex, particleIndex, sdfMagnitude, &collisionCount, stablizationPass,
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
      delta += boundaryCollision(&currentParticle, &selfParticleDiff, &collisionData, systemSettings, stablizationPass,
#ifdef MARK_COLLIDED_PARTICLES
        &particleCollisionData[particleIndex],
#endif
        &sharedData);

      if (collisionCount)
      {
        delta /= collisionCount;
      }

      // update position
      particlesPredictedNew[particleIndex].position += delta;
      particlesPredictedNew[particleIndex].identity = identity;

      if (stablizationPass)
      {
        particlesInit[particleIndex].position += delta;
        particlesInit[particleIndex].identity = identity;
      }

#ifdef MARK_COLLIDED_PARTICLES
      particleCollisionData[particleIndex].radius = fabs(collisionData.radius) * (collided ? -1.f : 1.f);
#endif
    }
  }
}

/*
@kernel Resolve particle collisions.
@param gridCellParticleOffsets Starting offset for each grid cell.
@param gridCellIndexCount Particle count for each grid cell.
@param gridCellParticleIndices Output array for particle indices.
@param gridParticleCellIndex Computed cell index for each particle.
@param particlesPredictedNew Updated particle positions post collision processing.
@param particlesNew Integrated particle position.
@param particlesBufferOld Old particle position.
@param particleCollisionData Array containing particle SDF mass and radius data.
@param particleSharedData Particle entity shared data.
@param partitions Instance partition data.
@param entityLocation Entity section data.
@param systemSettings Settings for the physics system.
@param gridSize Size of grid in one dimension.
@param stablizationPass Marks if this is a stablization pass.
@param nodeCount Total nodes in the solver.
*/
Kernel void applyCollisionsPerParticle(
  const Device uint*                  gridCellParticleOffsets,
  const Device uint*                  gridCellIndexCount,
  const Device uint*                  gridCellParticleIndices,
  const Device uint*                  gridParticleCellIndex,
  Device ParticleStruct*              particlesPredictedNew,
  Device ParticleStruct*              particlesNew,
  const Device ParticleDifferential*  particlesDiff,
  const Device ParticleStruct*        particlesBufferOld,
#ifdef MARK_COLLIDED_PARTICLES
  Device ParticleCollisionData*       particleCollisionData,
#else
  const Device ParticleCollisionData* particleCollisionData,
#endif
  const Device ParticleSharedData*    particleSharedData,
  Const PhySystemSettings*            systemSettings,
  constantKernelInput(int,            gridSize),
  constantKernelInput(uint,           stablizationPass),
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
  short collisionCount = 0;

  // current particle data
  float3 positionDiff = constructFloat3(0.f);
  ParticleStruct selfParticle = particlesBufferOld[particleIndex];
  const IdentityInfo identity = selfParticle.identity;
  const ParticleDifferential selfParticleDiff = particlesDiff[particleIndex];

  float sdfMagnitude;

#ifdef MARK_COLLIDED_PARTICLES
  bool collided = false;
#endif

  ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(identity);
  const PhySystemOffsets phySystemOffsets = systemSettings->globalOffsets[nodeIdentity.solverType];

  nodeIdentity.entityId += phySystemOffsets.globalSolverOffset;
  nodeIdentity.instanceId += phySystemOffsets.globalInstanceOffset;

  const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];
  const ParticleCollisionData collisionData = particleCollisionData[particleIndex];

  sdfMagnitude = length(collisionData.transformedSdfGradient);

  const short3 particleGridCellIndex = constructShort3(
    gridCellIndex & (gridSize - 1),
    (gridCellIndex / gridSize) & (gridSize - 1),
    gridCellIndex / (gridSize * gridSize)
  );

  for (short k=-1; k<2; k++)
  {
    const short z = particleGridCellIndex.z + k;
    if (z < 0 || z >= gridSize)
    {
      continue;
    }
    for (short j=-1; j<2; j++)
    {
      const short y = particleGridCellIndex.y + j;
      if (y < 0 || y >= gridSize)
      {
        continue;
      }
      for (short i=-1; i<2; i++)
      {
        const short x = particleGridCellIndex.x + i;
        if (x < 0 || x >= gridSize)
        {
          continue;
        }
        const int gridCellIndex = x + gridSize * (y + z * gridSize);
        int count = gridCellIndexCount[gridCellIndex];

        if (count == 0)
        {
          continue;
        }

        const int end = gridCellParticleOffsets[gridCellIndex];

        // batchwise iterate over indices in the cell
        for (int otherIndex = end - count; otherIndex < end; otherIndex++)
        {
          // iterate over each particle in the loaded batch
          const int otherNodeIndex = gridCellParticleIndices[otherIndex];

          const ParticleStruct otherParticle = particlesBufferOld[otherNodeIndex];
          const ParticleDifferential otherParticleDiff = particlesDiff[otherNodeIndex];
          positionDiff += sharedData.collisionDamping * processParticleCollision(&selfParticle, &selfParticleDiff, &otherParticle, &otherParticleDiff,
            &collisionData, &sharedData, otherNodeIndex, particleIndex, sdfMagnitude, &collisionCount, stablizationPass,
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
  positionDiff += boundaryCollision(&selfParticle, &selfParticleDiff, &collisionData, systemSettings, stablizationPass,
#ifdef MARK_COLLIDED_PARTICLES
    &particleCollisionData[particleIndex],
#endif
    &sharedData);

  if (collisionCount)
  {
    positionDiff /= collisionCount;
  }

  // update position
  particlesPredictedNew[particleIndex].position += positionDiff;
  particlesPredictedNew[particleIndex].identity = identity;

  if (stablizationPass)
  {
    particlesNew[particleIndex].position += positionDiff;
    particlesNew[particleIndex].identity = identity;
  }

#ifdef MARK_COLLIDED_PARTICLES
  particleCollisionData[particleIndex].radius = fabs(collisionData.radius) * (collided ? -1.f : 1.f);
#endif
}


#endif
