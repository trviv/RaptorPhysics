#ifndef COLLISION_SOLVER_SHADER
#define COLLISION_SOLVER_SHADER

//#define MARK_COLLIDED_PARTICLES

Kernel void assignMortonCodeKernel(
  Device BVHLeafInfo*           bvhLeafs,
  const Device ParticleStruct*  particles,
  const uint length)
{
  const uint index = threadIndex();

  if (index < length)
  {
    BVHLeafInfo bvhLeaf;
    bvhLeaf.mortonCode = get32BitMortonCode(particles + index);
    bvhLeaf.index = index;

    bvhLeafs[index] = bvhLeaf;
  }
}

uint2 quantizePosition(const float3 position, const uint gridSize)
{
  const float positionScale = 1.f;
  const float2 particlePredictedScaled = position.xy * positionScale + gridSize / 2;
  return clamp((uint2)(particlePredictedScaled.x, particlePredictedScaled.y), (uint2)(0, 0), (uint2)(gridSize - 1, gridSize - 1));
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
    const ParticleStruct predicted = particlesPredicted[index];
    const uint2 quantizedPosition = quantizePosition(predicted.position, gridSize);
    const uint gridCountOffset = quantizedPosition.y * gridSize + quantizedPosition.x;

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
  const uint                          nodeCount,
  const uint                          occupiedCellCount)
{
#define COLLISION_BATCH_SIZE 64

  Shared float2 sharedParticlesCollisionRadiusMass[COLLISION_BATCH_SIZE];
  Shared float3 sharedParticlesCollisionSdfGradient[COLLISION_BATCH_SIZE];
  Shared ParticleStruct sharedParticlesPredictedOld[COLLISION_BATCH_SIZE];

  const uint cellIndex = threadGroupIndex();
  const uint localIndex = threadLocalIndex();

  // within valid grid cell bounds
  if (cellIndex < occupiedCellCount)
  {
    // the valid cell to process
    const uint gridCellIndex = gridCompactCellIndices[cellIndex];

    // particle index buffer
    const int end = gridCellParticleOffsets[gridCellIndex];
    const int count = gridCellIndexCount[gridCellIndex];
    const int batchBegin = end - count + localIndex;
    const int batchEnd = ((end + COLLISION_BATCH_SIZE - 1) / COLLISION_BATCH_SIZE) * COLLISION_BATCH_SIZE;

    // batchwise iterate over indices in the cell
    for (uint baseIndex = batchBegin; baseIndex < batchEnd; baseIndex += COLLISION_BATCH_SIZE)
    {
      uint particleIndexBase;
      ParticleStruct outputBase;
      ParticleStruct predictedBase;
      ParticleCollisionData collisionDataBase;
      float sdfMagnitudeBase;
#ifdef MARK_COLLIDED_PARTICLES
      bool collided = false;
#endif

      if (baseIndex < end)
      { // read this batch
        particleIndexBase = gridCellParticleIndices[baseIndex];
        ParticleStruct predicted = particlesPredictedOld[particleIndexBase];

        ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(predicted.identity);
        const PhySystemOffsets phySystemOffsets = globalOffsets[nodeIdentity.solverType];

        nodeIdentity.entityId += phySystemOffsets.globalSolverOffset;
        nodeIdentity.instanceId += phySystemOffsets.globalInstanceOffset;

        const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];
        const ParticleCollisionData collisionData = getSDFUsingDeviceCollision(&sharedData, particleCollisionData, particleIndexBase);

        outputBase = predicted;
        predictedBase = predicted;
        collisionDataBase = collisionData;
        sdfMagnitudeBase = length(collisionData.transformedSdfGradient);
      }

      // batchwise iterate over indices in the cell
      for (int otherIndex = batchBegin; otherIndex < batchEnd; otherIndex += COLLISION_BATCH_SIZE)
      {
        // if both batches are different
        if (otherIndex != baseIndex)
        {
          if (otherIndex < end)
          {
            const uint particleIndex = gridCellParticleIndices[otherIndex];
            ParticleStruct predicted = particlesPredictedOld[particleIndex];

            ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(predicted.identity);
            const PhySystemOffsets phySystemOffsets = globalOffsets[nodeIdentity.solverType];

            nodeIdentity.entityId += phySystemOffsets.globalSolverOffset;
            nodeIdentity.instanceId += phySystemOffsets.globalInstanceOffset;

            const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];
            const ParticleCollisionData collisionData = getSDFUsingDeviceCollision(&sharedData, particleCollisionData, particleIndex);

            sharedParticlesPredictedOld[localIndex] = predicted;
            sharedParticlesCollisionRadiusMass[localIndex] = (float2)(collisionData.radius, collisionData.invMass);
            sharedParticlesCollisionSdfGradient[localIndex] = collisionData.transformedSdfGradient;
          }
        }
        else
        {
          // if same batches, copy from local data
          sharedParticlesPredictedOld[localIndex] = predictedBase;
          sharedParticlesCollisionRadiusMass[localIndex] = (float2)(collisionDataBase.radius, collisionDataBase.invMass);
          sharedParticlesCollisionSdfGradient[localIndex] = collisionDataBase.transformedSdfGradient;
        }

        localMemBarrier();

        // iterate over each particle in the loaded batch
        for (int j = 0; j < min((int)COLLISION_BATCH_SIZE, ((end - otherIndex) / COLLISION_BATCH_SIZE) * COLLISION_BATCH_SIZE); j++)
        {
          const ParticleStruct predicted = sharedParticlesPredictedOld[j];

          // skip if the base and the batch particle are of the same object
          if (predicted.identity.identity != predictedBase.identity.identity)
          {
            const float2 radiusMass = sharedParticlesCollisionRadiusMass[j];
            const float3 distanceVector = predictedBase.position - predicted.position;
            const float actualDistance = dot(distanceVector, distanceVector);

#ifdef MARK_COLLIDED_PARTICLES
            const float allowedDistance = sqr(fabs(collisionDataBase.radius) + fabs(radiusMass.x));
#else
            const float allowedDistance = sqr(collisionDataBase.radius + radiusMass.x);
#endif
            // if overlapping
            if (actualDistance < allowedDistance)
            {
              const float3 transformedSdfGradient = sharedParticlesCollisionSdfGradient[j];
              const float sdfMagnitude2 = length(transformedSdfGradient);

              float3 normal = (sdfMagnitudeBase < sdfMagnitude2) ? collisionDataBase.transformedSdfGradient : -transformedSdfGradient;
              float3 delta = normal * (collisionDataBase.invMass / (collisionDataBase.invMass + radiusMass.y));
              outputBase.position -= delta;
#ifdef MARK_COLLIDED_PARTICLES
              collided = true;
#endif
            }
          }
        }

        localMemBarrier();
      }

      if (baseIndex < end)
      {
        particlesPredictedNew[particleIndexBase] = outputBase;
#ifdef MARK_COLLIDED_PARTICLES
        particleCollisionData[particleIndexBase].radius = fabs(collisionDataBase.radius) * (collided ? -1.f : 1.f);
#endif
      }
    }
  }
}

/*
@kernel Apply boundary constrain.
@param particles Initial particle position.
@param particlesPredicted Integrated particle position.
@param particleSharedData Particle entity shared data.
@param particleAuxData Additional particle data.
@param partitions Instance partition data.
@param entityLocation Buffer containing entity boundary info.
@param globalOffsets Offsets to particle nodes all the solvers.
@param nodeCount Total nodes in the solver.
*/
Kernel void boundaryCollisionKernel(
  Device ParticleStruct*              particles,
  Device ParticleStruct*              particlesPredicted,
  const Device ParticleCollisionData* particleCollisionData,
  const Device ParticleSharedData*    particleSharedData,
  const Device ParticleAuxData*       particleAuxData,
  const Device PartitionInfo*         partitions,
  const Device EntityLocation*        entityLocation,
  Const PhySystemOffsets*             globalOffsets,
  const uint                          nodeCount)
{
  const uint index = threadIndex();

  if (index < nodeCount)
  {
    ParticleStruct particle = particles[index];
    ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(particle.identity);

    const PhySystemOffsets phySystemOffsets = globalOffsets[nodeIdentity.solverType];

    nodeIdentity.entityId += phySystemOffsets.globalSolverOffset;
    nodeIdentity.instanceId += phySystemOffsets.globalInstanceOffset;

    const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];
    const ParticleNodeLocator nodeLocator = getNodeLocator(index, phySystemOffsets.globalNodeOffset + partitions[nodeIdentity.instanceId].offset, entityLocation[nodeIdentity.entityId].node);

    //const ParticleAuxData auxData = particleAuxData[nodeLocator.commonNodeIndex];
    //const float invMass = getInvMassUsingThreadAux(&sharedData, &auxData);
    const ParticleCollisionData collisionData = getSDFUsingDeviceCollision(&sharedData, particleCollisionData, index);
    const float invMass = collisionData.invMass;

    if (invMass) // only if movable
    {
      float dely = 0.f;

      if (particlesPredicted[nodeLocator.absoluteNodeIndex].position.y <= -0.f)
      {
        dely = 0.f - particlesPredicted[nodeLocator.absoluteNodeIndex].position.y;

        particles[nodeLocator.absoluteNodeIndex].position.y += dely;
        particlesPredicted[nodeLocator.absoluteNodeIndex].position.y += dely;

        //float mag = length(collisionData.transformedSdfGradient);
        //if (fabs(dely) < mag)
        //{
        //  dely = mag;
        //}
        //if (mag < 0.00001f)
        //{
        //  mag = 1.f;
        //}
        //mag = 1.f;

        ////particles[nodeLocator.absoluteNodeIndex].position -= collisionData.transformedSdfGradient / mag * dely;
        ////particlesPredicted[nodeLocator.absoluteNodeIndex].position -= collisionData.transformedSdfGradient / mag * dely;
        //particles[nodeLocator.absoluteNodeIndex].position.y += 1.f / mag * dely;
        //particlesPredicted[nodeLocator.absoluteNodeIndex].position.y += 1.f / mag * dely;
      }
    }
  }
}

#endif