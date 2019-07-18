#ifndef COLLISION_SOLVER_SHADER
#define COLLISION_SOLVER_SHADER

uint3 quantizePosition(const float3 position, const uint gridSize)
{
  const float3 particlePredictedScaled = fabs(position.xyz);
  return constructUint3(particlePredictedScaled.x, particlePredictedScaled.y, particlePredictedScaled.z) & constructUint3(gridSize - 1, gridSize - 1, gridSize - 1);
}

/*
@kernel Compute and store the cell index for each particle, and atomically increment the cell count for grid cell.
@param gridCellIndexCount Particle count for each grid cell.
@param gridParticleCellIndex Computed cell index for each particle.
@param particlesPredicted Integrated particle position.
@param partitions Instance partition data.
@param entityLocation Entity section data.
@param globalOffsets Offsets to particle nodes all the solvers.
@param nodeCount Total nodes in the solver.
@param gridSize Size of grid in one dimension.
*/
Kernel void createGridCellHistogram(
  Device uint*                      gridCellIndexCount,
  Device uint*                      gridParticleCellIndex,
  const Device ParticleStruct*      particlesPredicted,
  const Device PartitionInfo*       partitions,
  const Device EntityLocation*      entityLocation,
  Const PhySystemOffsets*           globalOffsets,
  const uint                        nodeCount,
  const uint                        gridSize)
{
  const uint index = threadIndex();

  if (index < nodeCount)
  {
    const uint3 quantizedPosition = quantizePosition(particlesPredicted[index].position, gridSize);
    const uint gridCountOffset = (quantizedPosition.z * gridSize + quantizedPosition.y) * gridSize + quantizedPosition.x;

    gridParticleCellIndex[index] = gridCountOffset;
    atomicAdd(gridCellIndexCount + gridCountOffset, 1);
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
  Device uint*                      gridCellParticleOffsets,
  const Device uint*                gridParticleCellIndex,
  const uint                        nodeCount)
{
  const uint index = threadIndex();

  if (index < nodeCount)
  {
    const uint gridCountOffset = gridParticleCellIndex[index];
    const uint offset = atomicAdd(gridCellParticleOffsets + gridCountOffset, 1);

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
@param particleAuxData Additional particle data.
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
  const Device ParticleStruct*        particlesPredictedOld,
#ifdef MARK_COLLIDED_PARTICLES
  Device ParticleCollisionData*       particleCollisionData,
#else
  const Device ParticleCollisionData* particleCollisionData,
#endif
  const Device ParticleSharedData*    particleSharedData,
  const Device ParticleAuxData*       particleAuxData,
  const Device PartitionInfo*         partitions,
  const Device EntityLocation*        entityLocation,
  Const PhySystemOffsets*             globalOffsets,
  const int                           occupiedCellCount)
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
    ParticleStruct output;
    IdentityInfo identity;
    ParticleStruct currentParticle;
    ParticleCollisionData collisionData;
    float sdfMagnitude;

#ifdef MARK_COLLIDED_PARTICLES
    bool collided = false;
#endif

    if (baseIndex < end)
    { // read this batch
      particleIndex = gridCellParticleIndices[baseIndex];
      currentParticle = particlesPredictedOld[particleIndex];
      identity = currentParticle.identity;

      ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(identity);
      const PhySystemOffsets phySystemOffsets = globalOffsets[nodeIdentity.solverType];

      nodeIdentity.entityId += phySystemOffsets.globalSolverOffset;
      nodeIdentity.instanceId += phySystemOffsets.globalInstanceOffset;

      const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];
      collisionData = getSDFUsingDeviceCollision(&sharedData, particleCollisionData, particleIndex);

      sdfMagnitude = length(collisionData.transformedSdfGradient);
    }

    // batchwise iterate over indices in the cell
    for (int otherIndex = end - count; otherIndex < end; otherIndex++)
    {
      // iterate over each particle in the loaded batch
      const int particleIndex = gridCellParticleIndices[otherIndex];
      const ParticleStruct predicted2 = particlesPredictedOld[particleIndex];

      ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(predicted2.identity);
      const PhySystemOffsets phySystemOffsets = globalOffsets[nodeIdentity.solverType];

      nodeIdentity.entityId += phySystemOffsets.globalSolverOffset;
      nodeIdentity.instanceId += phySystemOffsets.globalInstanceOffset;

      const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];
      const ParticleCollisionData collisionData2 = getSDFUsingDeviceCollision(&sharedData, particleCollisionData, particleIndex);

      currentParticle.position -= sharedData.collisionDamping * processParticleCollision(currentParticle, particlesPredictedOld[particleIndex], collisionData, particleIndex, sdfMagnitude,
#ifdef MARK_COLLIDED_PARTICLES
        particleCollisionData, &collided, &collisionCount);
#else
        particleCollisionData, &collisionCount);
#endif
      // apply boundary
      boundaryCollision(&currentParticle, 0, ParticleNodeLocator(), collisionData);
    }

    if (baseIndex < end)
    {
      currentParticle.identity = identity;
      particlesPredictedNew[particleIndex] = currentParticle;
#ifdef MARK_COLLIDED_PARTICLES
      particleCollisionData[particleIndex].radius = fabs(collisionData.radius) * (collided ? -1.f : 1.f);
#endif
    }
  }
}

#endif
