#ifndef GRID_SOLVER_SHADER
#define GRID_SOLVER_SHADER

//#define GRID_SOLVER_SEPARATE_LOOPS
#define GRID_SOLVER_HASH_FUNCTION

inline uint gridIndexInt3Int(const int3 relativeIndex, const int gridSize)
{
  return mad24(mad24(relativeIndex.z, gridSize, relativeIndex.y), gridSize, relativeIndex.x);
}

inline int3 positionHashFunction(const float3 position, const int gridSize)
{
  const int3 quantizedPosition = convertInt3(position) + gridSize;
  const int3 multiplier = quantizedPosition / gridSize - 1;
  return mad24(mad24(multiplier.zxy, 3, multiplier.yzx), 5, quantizedPosition) & constructInt3(gridSize - 1);
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
  const Device ParticleStruct*      particles,
  Const XAB*                        systemBoundingBox,
  Const float*                      invRadius,
  constantKernelInput(uint,         nodeCount),
  constantKernelInput(int,          gridSize)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint index = threadIndex();

  if (index < nodeCount)
  {
#ifndef GRID_SOLVER_HASH_FUNCTION
    const float3 inverseMergedBoxSize = ((float)gridSize) / max(gridSize / invRadius[0], systemBoundingBox->max - systemBoundingBox->min);
    const int3 quantizedPosition = convertInt3((particles[index].position - systemBoundingBox->min) * inverseMergedBoxSize);
#else
    const int3 quantizedPosition = positionHashFunction((particles[index].position - systemBoundingBox->min) * invRadius[0], gridSize);
#endif
    const uint gridCountOffset = gridIndexInt3Int(quantizedPosition, gridSize);

    gridParticleCellIndex[index] = gridCountOffset;

#ifdef USE_SIMD_COMPUTE
    const bool simdWriteToSame = simdAll(simdFirst(gridCountOffset) == gridCountOffset);

    if (simdWriteToSame)
    {
      const uint activeSimdThreads = simdReduce(1);
      if (simdIsFirst())
      {
        atomicAdd(&gridCellIndexCount[gridCountOffset], activeSimdThreads);
      }
    }
    else
#endif
    {
      atomicAdd(&gridCellIndexCount[gridCountOffset], 1);
    }
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

inline uchar encodeCellOffset(short x, short y, short z)
{
  x++, y++, z++;
  return (z << 4) | (y << 2) | x;
}

inline uint decodeCellIndex(uchar encodedOffset, short3 baseIndex, int gridSize)
{
  baseIndex--;
  return baseIndex.x + (encodedOffset & 3) + (baseIndex.y + ((encodedOffset >> 2) & 3) + (baseIndex.z + (encodedOffset >> 4)) * gridSize) * gridSize;
}

inline short3 decodeCellVector(uchar encodedOffset)
{
  return constructShort3(encodedOffset, (encodedOffset >> 2), (encodedOffset >> 4)) & constructShort3(3);
}

#ifdef GRID_SOLVER_HASH_FUNCTION

#define GRID_SOLVER_NEIGHBOUR_LOOP_BEGIN \
  for (short k=-1; k<2; k++) \
  { \
    for (short j=-1; j<2; j++) \
    { \
      for (short i=-1; i<2; i++) \
      { \
        const int3 quantizedPosition = positionHashFunction(particleCellPosition + constructFloat3(i, j, k), gridSize); \
        const int gridCellIndex = gridIndexInt3Int(quantizedPosition, gridSize);

#else

#define GRID_SOLVER_NEIGHBOUR_LOOP_BEGIN \
  for (short k=-1; k<2; k++) \
  { \
    const short z = particleGridCellIndex.z + k; \
    if (z < 0 || z >= gridSize) \
    { \
      continue; \
    } \
    for (short j=-1; j<2; j++) \
    { \
      const short y = particleGridCellIndex.y + j; \
      if (y < 0 || y >= gridSize) \
      { \
        continue; \
      } \
      for (short i=-1; i<2; i++) \
      { \
        const short x = particleGridCellIndex.x + i; \
        if (x < 0 || x >= gridSize) \
        { \
          continue; \
        } \
        const int gridCellIndex = gridIndexInt3Int(constructInt3(x, y, z), gridSize);

#endif

#define GRID_SOLVER_NEIGHBOUR_LOOP_END \
      } \
    } \
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
Kernel void applyCollisions(
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
  Const XAB*                          systemBoundingBox,
  Const float*                        invRadius,
  constantKernelInput(int,            gridSize),
  constantKernelInput(uint,           stablizationPass),
  constantKernelInput(uint,           nodeCount)
  KERNEL_GLOBAL_ARGUMENTS)
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

  sdfMagnitude = collisionData.gradientMagnitude;

  const short3 particleGridCellIndex = constructShort3(
    gridCellIndex & (gridSize - 1),
    (gridCellIndex / gridSize) & (gridSize - 1),
    gridCellIndex / mul24(gridSize, gridSize)
  );

#ifdef GRID_SOLVER_HASH_FUNCTION
  const float3 particleCellPosition = (selfParticle.position - systemBoundingBox->min) * invRadius[0];
#endif

#ifndef GRID_SOLVER_SEPARATE_LOOPS
  GRID_SOLVER_NEIGHBOUR_LOOP_BEGIN
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
  GRID_SOLVER_NEIGHBOUR_LOOP_END
#else
  uchar validNeighbourIndex[27];
  uchar validNeighbourCount = 0;

  GRID_SOLVER_NEIGHBOUR_LOOP_BEGIN
    int count = gridCellIndexCount[gridCellIndex];

    if (count == 0)
    {
      continue;
    }

    validNeighbourIndex[validNeighbourCount++] = encodeCellOffset(i, j, k);
  GRID_SOLVER_NEIGHBOUR_LOOP_END

  for (uchar i=0; i<validNeighbourCount; i++)
  {
#ifndef GRID_SOLVER_HASH_FUNCTION
    const int gridCellIndex = decodeCellIndex(validNeighbourIndex[i], particleGridCellIndex, gridSize);
#else
    const int3 quantizedPosition = positionHashFunction(particleCellPosition + constructFloat3(decodeCellVector(validNeighbourIndex[i]) - constructShort3(1)), gridSize);
    const int gridCellIndex = (quantizedPosition.z * gridSize + quantizedPosition.y) * gridSize + quantizedPosition.x;
#endif
    int count = gridCellIndexCount[gridCellIndex];

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

#endif

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
