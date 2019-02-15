#ifndef COLLISION_SOLVER_SHADER
#define COLLISION_SOLVER_SHADER

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
    ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(predicted.identity);
    const PhySystemOffsets phySystemOffsets = globalOffsets[nodeIdentity.solverType];

    nodeIdentity.entityId += phySystemOffsets.globalSolverOffset;
    nodeIdentity.instanceId += phySystemOffsets.globalInstanceOffset;

    const ParticleNodeLocator nodeLocator = getNodeLocator(index, phySystemOffsets.globalNodeOffset + partitions[nodeIdentity.instanceId].offset, entityLocation[nodeIdentity.entityId].node);

    const float positionScale = 4.f;
    const float2 particlePredictedScaled = predicted.position.xy * positionScale + gridSize / 2;
    const uint2 particlePredictedPos = clamp((uint2)(particlePredictedScaled.x, particlePredictedScaled.y), (uint2)(0, 0), (uint2)(gridSize - 1, gridSize - 1));
    const uint gridCountOffset = particlePredictedPos.y * gridSize + particlePredictedPos.x;

    gridParticleCellIndex[index] = gridCountOffset;
    atomicAdd(gridCellIndexCount + gridCountOffset, 1);
  }
}


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


Kernel void applyCollisions(
  const Device uint*                gridCellParticleIndices,
  const Device uint*                gridCellParticleCount,
  const Device uint*                gridCellParticleOffsets,
  const Device uint*                gridParticleCellIndex,
  const Device uint*                gridCompactCellIndices,
  Device ParticleStruct*            particlesPredicted,
  const Device IdentityInfo*        particleIdentities,
  const Device PartitionInfo*       partitions,
  const Device EntityLocation*      entityLocation,
  Const PhySystemOffsets*           globalOffsets,
  const uint                        nodeCount,
  const uint                        occupiedCellCount)
{
  const uint cellIndex = threadIndex();

  if (cellIndex < occupiedCellCount)
  {
    /*const uint gridCellIndex = gridCompactCellIndices[cellIndex];

    uint count = gridCellParticleCount[gridCellIndex];
    uint end = gridCellParticleOffsets[gridCellIndex];
    uint start = end + -count;

    for (uint i = start; i < end; i++)
    {
    gridCellParticleIndices[i];
    }

    //const uint index = ;

    const IdentityInfo identity = particleIdentities[index];
    ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(identity);
    const PhySystemOffsets phySystemOffsets = globalOffsets[nodeIdentity.solverType];

    nodeIdentity.entityId += phySystemOffsets.globalSolverOffset;
    nodeIdentity.instanceId += phySystemOffsets.globalInstanceOffset;

    const ParticleNodeLocator nodeLocator = getNodeLocator(index, phySystemOffsets.globalNodeOffset + partitions[nodeIdentity.instanceId].offset, entityLocation[nodeIdentity.entityId].node);*/

    /*const float positionScale = 4.f;
    const float2 particlePredictedScaled = particlesPredicted[nodeLocator.absoluteNodeIndex].position.xy * positionScale + gridSize / 2;
    const uint2 particlePredictedPos = clamp((uint2)(particlePredictedScaled.x, particlePredictedScaled.y), (uint2)(0, 0), (uint2)(gridSize - 1, gridSize - 1));
    const uint gridCountOffset = particlePredictedPos.y * gridSize + particlePredictedPos.x;

    gridParticleCellIndex[index] = gridCountOffset;
    atomicAdd(gridCellIndexCount + gridCountOffset, 1);*/
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

    const ParticleAuxData auxData = particleAuxData[nodeLocator.commonNodeIndex];
    const float invMass = getInvMassUsingThreadAux(&sharedData, &auxData);
    const ParticleCollisionData collisionData = getSDFUsingDeviceCollision(&sharedData, particleCollisionData, index);

    if (invMass) // only if movable
    {
      float dely = 0.f;

      if (particlesPredicted[nodeLocator.absoluteNodeIndex].position.y <= -0.f)
      {
        dely = -0.f - particlesPredicted[nodeLocator.absoluteNodeIndex].position.y;

        //particle.position.y += dely;
        //particles[nodeLocator.absoluteNodeIndex].position.y = particle.position.y;
        //particlesPredicted[nodeLocator.absoluteNodeIndex].position.y += dely;

        if (fabs(dely) < collisionData.sdfMagnitude)
        {
          dely = collisionData.sdfMagnitude;
        }

        particles[nodeLocator.absoluteNodeIndex].position -= collisionData.transformedSdfGradient * dely;
        particlesPredicted[nodeLocator.absoluteNodeIndex].position -= collisionData.transformedSdfGradient * dely;
      }
    }
  }
}

#endif