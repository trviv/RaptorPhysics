#ifndef GRID_SOLVER_SHADER
#define GRID_SOLVER_SHADER

//#define GRID_SOLVER_SEPARATE_LOOPS
//#define DEBUG_GRID_SOLVER_SCATTER

inline uint gridIndexInt3Int(const int3 relativeIndex, const int gridSizeExp)
{
  return ((((relativeIndex.z << gridSizeExp) + relativeIndex.y) << gridSizeExp) + relativeIndex.x);
}

inline int3 positionHashFunction(const float3 position, const int gridSize, const int gridSizeExp)
{
  const int3 quantizedPosition = convertInt3(position) + gridSize - 1;
  const int3 multiplier = (quantizedPosition >> gridSizeExp);
  // TODO: Find a hash function which does not have a possbility of collision
  return mad24(mad24(multiplier.zxy, 3, multiplier.yzx), 5, quantizedPosition) & constructInt3(gridSize - 1);
}

inline uint encodeScatterCellIndex(const uint cellIndex, const ushort serialIndex)
{
#ifdef DEBUG_GRID_SOLVER_SCATTER
  return cellIndex + (serialIndex * 1000000);
#else
  return cellIndex | (serialIndex << 29);
#endif
}

inline void decodeScatterCellIndex(Thread uint* cellIndex, Thread ushort* serialIndex, const uint encodedCellIndex)
{
#ifdef DEBUG_GRID_SOLVER_SCATTER
  *serialIndex = encodedCellIndex / 1000000;
  *cellIndex = encodedCellIndex % 1000000;
#else
  *serialIndex = encodedCellIndex >> 29;
  *cellIndex = encodedCellIndex & 0x1FFFFFFF;
#endif
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
#ifdef GRID_COLLISION_SOLVER_SCATTER_PARTICLES
  const Device ParticleSharedData*  particleSharedData,
  const Device ParticleAuxData*     particleAuxData,
  const Device PartitionInfo*       partitions,
  const Device EntityLocation*      entityLocation,
#ifdef COLLISION_SOLVER_USE_SYSTEM_OFFSETS
  Const PhySystemSettings*          systemSettings,
#endif
#endif
  Const XAB*                        systemBoundingBox,
  Const float*                      invRadius,
  constantKernelInput(uint,         nodeCount),
  constantKernelInput(int,          gridSize),
  constantKernelInput(int,          gridSizeExp)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint index = threadIndex();

  if (index < nodeCount)
  {
#ifndef GRID_COLLISION_SOLVER_SCATTER_PARTICLES

#ifndef GRID_SOLVER_HASH_FUNCTION
    const float3 inverseMergedBoxSize = ((float)gridSize) / max(gridSize / invRadius[0], systemBoundingBox->max - systemBoundingBox->min);
    const int3 quantizedPosition = convertInt3((particles[index].position - systemBoundingBox->min) * inverseMergedBoxSize);
#else
    const int3 quantizedPosition = positionHashFunction((particles[index].position - systemBoundingBox->min) * invRadius[0], gridSize, gridSizeExp);
#endif
    const uint gridCountOffset = gridIndexInt3Int(quantizedPosition, gridSizeExp);

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

#else

    // get radius
    const ParticleStruct particle = particles[index];
    ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(particle.identity);
#ifdef COLLISION_SOLVER_USE_SYSTEM_OFFSETS
    const PhySystemOffsets phySystemOffsets = systemSettings->globalOffsets[nodeIdentity.solverType];

    nodeIdentity.entityId += phySystemOffsets.globalSolverOffset;
    nodeIdentity.instanceId += phySystemOffsets.globalInstanceOffset;

    const ParticleNodeLocator nodeLocator = getNodeLocator(index, phySystemOffsets.globalNodeOffset + partitions[nodeIdentity.instanceId].offset, entityLocation[nodeIdentity.entityId].node);
#else
    const ParticleNodeLocator nodeLocator = getNodeLocator(index, partitions[nodeIdentity.instanceId].offset, entityLocation[nodeIdentity.entityId].node);
#endif

    const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];

    const float radius = getRadiusUsingDeviceAux(&sharedData, particleAuxData, nodeLocator.commonNodeIndex);

    uint gridCellIndices[9];

    float3 centerDeltas[] = {
      { 0.f,  0.f,  0.f},
      {-1.f, -1.f, -1.f},
      { 1.f, -1.f, -1.f},
      {-1.f,  1.f, -1.f},
      { 1.f,  1.f, -1.f},
      {-1.f, -1.f,  1.f},
      { 1.f, -1.f,  1.f},
      {-1.f,  1.f,  1.f},
      { 1.f,  1.f,  1.f}
    };

    const float3 center = (particle.position - systemBoundingBox->min);

    for (short i=0; i<9; i++)
    {
#ifndef GRID_SOLVER_HASH_FUNCTION
      const float3 inverseMergedBoxSize = ((float)gridSize) / max(gridSize / invRadius[0], systemBoundingBox->max - systemBoundingBox->min);
      const int3 quantizedPosition = convertInt3((center + radius * centerDeltas[i]) * inverseMergedBoxSize);
#else
      const int3 quantizedPosition = positionHashFunction((center + radius * centerDeltas[i]) * invRadius[0], gridSize, gridSizeExp);
#endif
      gridCellIndices[i] = gridIndexInt3Int(quantizedPosition, gridSizeExp);
    }

    for (short i=0; i<9; i++)
    {
      for (short j=i+1; j<9; j++)
      {
        gridCellIndices[j] = select(gridCellIndices[j], (uint)-1, gridCellIndices[i] == gridCellIndices[j]);
      }
    }

    for (short i=1; i<9; i++)
    {
      for (short j=i+1; j<9; j++)
      {
        if (gridCellIndices[i] > gridCellIndices[j])
        {
          const uint temp = gridCellIndices[i];
          gridCellIndices[i] = gridCellIndices[j];
          gridCellIndices[j] = temp;
        }
      }
    }

    ((Device commonUint8*)gridParticleCellIndex)[index] = *((Thread commonUint8*)gridCellIndices);

    for (short i=0; i<8; i++)
    {
      if (gridCellIndices[i] == -1)
        break;

      atomicAdd(&gridCellIndexCount[gridCellIndices[i]], 1);
    }
#endif
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
#ifndef GRID_COLLISION_SOLVER_SCATTER_PARTICLES
    const uint gridCountOffset = gridParticleCellIndex[index];
    const uint offset = atomicAdd(&gridCellParticleOffsets[gridCountOffset], 1);

    gridCellParticleIndices[offset] = index;
#else
    uint gridCountOffsets[8];
    *((Thread commonUint8*)gridCountOffsets) = ((const Device commonUint8*)gridParticleCellIndex)[index];

    for (short i=0; i<8; i++)
    {
      if (gridCountOffsets[i] == -1)
        break;

      const uint offset = atomicAdd(&gridCellParticleOffsets[gridCountOffsets[i]], 1);
      gridCellParticleIndices[offset] = encodeScatterCellIndex(index, i);
    }
#endif
  }
}

inline uchar encodeCellOffset(short x, short y, short z)
{
  x++, y++, z++;
  return (z << 4) | (y << 2) | x;
}

inline uint decodeCellIndex(uchar encodedOffset, short3 baseIndex, int gridSizeExp)
{
  baseIndex--;
  return baseIndex.x + (encodedOffset & 3) + ((baseIndex.y + ((encodedOffset >> 2) & 3) + ((baseIndex.z + (encodedOffset >> 4)) << gridSizeExp)) << gridSizeExp);
}

inline short3 decodeCellVector(uchar encodedOffset)
{
  return constructShort3(encodedOffset, (encodedOffset >> 2), (encodedOffset >> 4)) & constructShort3(3);
}

#ifdef GRID_SOLVER_HASH_FUNCTION

#define GRID_SOLVER_NEIGHBOUR_LOOP_BEGIN \
  short x = -2; \
  short y = -1; \
  short z = -1; \
  for (short n=0; n<27; n++) \
  { \
    x++; \
    if (x == 2) \
    { y++; x = -1;} \
    if (y == 2) \
    { z++; y = -1;} \
    const int3 quantizedPosition = positionHashFunction(particleCellPosition + constructFloat3(x, y, z), gridSize, gridSizeExp); \
    const int gridCellIndex = gridIndexInt3Int(quantizedPosition, gridSizeExp);

#else

#define GRID_SOLVER_NEIGHBOUR_LOOP_BEGIN \
  short x = particleGridCellIndex.x - 2; \
  short y = particleGridCellIndex.y - 1; \
  short z = particleGridCellIndex.z - 1; \
  const short maxX = particleGridCellIndex.x + 2; \
  const short maxY = particleGridCellIndex.y + 2; \
  for (short n=0; n<27; n++) \
  { \
    x++; \
    if (x == maxX) \
    { y++; x -= 3;} \
    if (y == maxY) \
    { z++; y -= 3;} \
    if ((x < 0 | x >= gridSize) | (y < 0 | y >= gridSize) | (z < 0 | z >= gridSize))  \
    { \
      continue; \
    } \
    const int gridCellIndex = gridIndexInt3Int(constructInt3(x, y, z), gridSizeExp);

#endif

#define GRID_SOLVER_NEIGHBOUR_LOOP_END }

// function to get previous and current offset, previous will be the starting and current will be the end index
uint2 getRangeFromOffset(const Device uint* gridCellParticleOffsets, const uint gridCellIndex)
{
  const uint2 ret = *((const Device uint2*)(gridCellParticleOffsets + gridCellIndex + select(0, -1, gridCellIndex)));
  return select(constructUint2(0, ret.x), ret, selectInput2(gridCellIndex));
}

/*
@kernel Resolve particle collisions.
@param gridCellParticleOffsets Starting offset for each grid cell.
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
  constantKernelInput(int,            gridSizeExp),
  constantKernelInput(uint,           stablizationPass),
  constantKernelInput(uint,           nodeCount)
#ifdef GRID_COLLISION_SOLVER_USE_SHARED_MEMORY
  , sharedMemKernelInput(CollisionSharedData, localCollisionSharedData, 16)
  , sharedMemKernelInput(ParticleStruct,      localOtherParticle,       17)
  , sharedMemKernelInput(uint2,               localTestQueue,           18)
#endif
  KERNEL_GLOBAL_ARGUMENTS
  KERNEL_THREAD_ARGUMENTS)
{
  uint particleIndex = threadIndex();
  const ushort localIndex = threadLocalIndex();

#ifdef GRID_COLLISION_SOLVER_SCATTER_PARTICLES
  if (particleIndex >= gridCellParticleOffsets[(1 << (3*gridSizeExp)) - 1])
  {
    return;
  }
#else
  if (particleIndex >= nodeCount)
  {
    return;
  }
#endif

  particleIndex = gridCellParticleIndices[particleIndex];

#ifndef GRID_COLLISION_SOLVER_SCATTER_PARTICLES

  const uint gridCellIndex = gridParticleCellIndex[particleIndex];

  // current particle data
  ParticleStruct selfParticle = particlesBufferOld[particleIndex];
  const IdentityInfo identity = selfParticle.identity;

  ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(identity);
  const PhySystemOffsets phySystemOffsets = systemSettings->globalOffsets[nodeIdentity.solverType];

  nodeIdentity.entityId += phySystemOffsets.globalSolverOffset;
  nodeIdentity.instanceId += phySystemOffsets.globalInstanceOffset;

#ifdef GRID_COLLISION_SOLVER_USE_SHARED_MEMORY
  CollisionSharedData collisionSharedData;

  collisionSharedData.sharedCollisionSolverData = particleSharedData[nodeIdentity.entityId].collisionSolverData;
  collisionSharedData.sharedCollisionData = particleCollisionData[particleIndex];
  collisionSharedData.sharedSelfParticle = selfParticle;
  collisionSharedData.sharedSelfParticleDiff = particlesDiff[particleIndex];
  collisionSharedData.sharedParticleIndex = particleIndex;
  collisionSharedData.sharedSolverType = getSolverType(identity);
  collisionSharedData.sharedPositionDiff = 0.f;
  collisionSharedData.sharedCollisionCount = 0;

  localCollisionSharedData[localIndex] = collisionSharedData;

#define collisionSolverData collisionSharedData.sharedCollisionSolverData
#define collisionData       collisionSharedData.sharedCollisionData
#define selfParticle        collisionSharedData.sharedSelfParticle
#define selfParticleDiff    collisionSharedData.sharedSelfParticleDiff
#define positionDiff        collisionSharedData.sharedPositionDiff
#define collisionCount      collisionSharedData.sharedCollisionCount
#define solverType          collisionSharedData.sharedSolverType

  Shared int pendingTestCount;
  Shared int doneTestCount;

  if (localIndex == 0)
  {
    pendingTestCount = 0;
    doneTestCount = 0;
  }
  localMemBarrier();

  atomicAddShared(&doneTestCount, 1);
  localMemBarrier();

  // TODO: revisit logic to find out how to calculate this
  const short workgroupThreads = doneTestCount;
  //min(nodeCount - (particleIndex/ComputeSimdWidth)*ComputeSimdWidth, (uint)ComputeSimdWidth);
#else
  const CollisionSolverData collisionSolverData = particleSharedData[nodeIdentity.entityId].collisionSolverData;
  const ParticleCollisionData collisionData = particleCollisionData[particleIndex];
  const ParticleDifferential selfParticleDiff = particlesDiff[particleIndex];
  float3 positionDiff = constructFloat3(0.f);
  uint collisionCount = 0;
  const ushort solverType = getSolverType(identity);
#endif

  const short3 particleGridCellIndex = constructShort3(
    gridCellIndex & (gridSize - 1),
    (gridCellIndex >> gridSizeExp) & (gridSize - 1),
    gridCellIndex >> (gridSizeExp << 1)
  );

#ifdef GRID_SOLVER_HASH_FUNCTION
  const float3 particleCellPosition = (selfParticle.position - systemBoundingBox->min) * invRadius[0];
#endif

#ifndef GRID_SOLVER_SEPARATE_LOOPS
  GRID_SOLVER_NEIGHBOUR_LOOP_BEGIN
#ifdef GRID_COLLISION_SOLVER_USE_SHARED_MEMORY
    //localMemBarrier();
    if (localIndex == 0)
    {
      doneTestCount = 0;
    }
    localMemBarrier();

    const uint2 indexRange = getRangeFromOffset(gridCellParticleOffsets, gridCellIndex);
    bool cellDone = (indexRange.x == indexRange.y);
    if (cellDone)
    {
      atomicAddShared(&doneTestCount, 1);
    }
    localMemBarrier();

    // batchwise iterate over indices in the cell
    for (int otherIndex = indexRange.x; doneTestCount < workgroupThreads | pendingTestCount > 0; )
    {
      const short threadTestCount = pendingTestCount;
      if (threadTestCount >= workgroupThreads | (doneTestCount == workgroupThreads & localIndex < threadTestCount))
      {
        const uint2 sharedDataIndex = localTestQueue[localIndex];
        const ParticleDifferential otherParticleDiff = particlesDiff[sharedDataIndex.x];
        uint collisions = 0;

        const CollisionSharedData collisionSharedData = localCollisionSharedData[sharedDataIndex.y];
        const float3 diff = processParticleCollision(&collisionSharedData.sharedSelfParticle, &collisionSharedData.sharedSelfParticleDiff, &localOtherParticle[localIndex], &otherParticleDiff, true, &collisionSharedData.sharedCollisionData, &collisionSharedData.sharedCollisionSolverData, sharedDataIndex.x, collisionSharedData.sharedParticleIndex, collisionSharedData.sharedCollisionData.gradientMagnitude, &collisions, stablizationPass, collisionSharedData.sharedSolverType, particlesPredictedNew, particleCollisionData);

        atomicAddFloat3Shared(&localCollisionSharedData[sharedDataIndex.y].sharedPositionDiff, diff);
        atomicAddShared(&localCollisionSharedData[sharedDataIndex.y].sharedCollisionCount, collisions);

        localMemBarrier();

        if (threadTestCount >= workgroupThreads)
        {
          localTestQueue[localIndex] = localTestQueue[workgroupThreads + localIndex];
          localOtherParticle[localIndex] = localOtherParticle[workgroupThreads + localIndex];
        }

        if (localIndex == 0)
        {
          pendingTestCount = select(0, pendingTestCount - workgroupThreads, threadTestCount >= workgroupThreads);
        }
      }
      localMemBarrier();

      if (!cellDone)
      {
        // iterate over each particle in the loaded batch
        const uint otherNodeIndex = gridCellParticleIndices[otherIndex];
        const ParticleStruct otherParticle = particlesBufferOld[otherNodeIndex];
        if (++otherIndex == indexRange.y)
        {
          cellDone = true;
          atomicAddShared(&doneTestCount, 1);
        }

        // basic check to determine if collision can happen between the particles
        if (shouldCheckForCollision(getSolverType(selfParticle.identity), particleIndex, otherNodeIndex, &selfParticle, &otherParticle))
        {
          // add to test list at an available offset
          const ushort testQueueIndex = atomicAddSignedShared(&pendingTestCount, 1);
          // store test info
          localTestQueue[testQueueIndex] = constructUint2(otherNodeIndex, localIndex);
          localOtherParticle[testQueueIndex] = otherParticle;
        }
      }
      localMemBarrier();
    }
#else
    const uint2 indexRange = getRangeFromOffset(gridCellParticleOffsets, gridCellIndex);

    // batchwise iterate over indices in the cell
    for (int otherIndex = indexRange.x; otherIndex < indexRange.y; otherIndex++)
    {
      // iterate over each particle in the loaded batch
      const int otherNodeIndex = gridCellParticleIndices[otherIndex];

      const ParticleStruct otherParticle = particlesBufferOld[otherNodeIndex];
      if (!shouldCheckForCollision(solverType, particleIndex, otherNodeIndex, &selfParticle, &otherParticle))
        continue;

      const ParticleDifferential otherParticleDiff = particlesDiff[otherNodeIndex];
      positionDiff += processParticleCollision(&selfParticle, &selfParticleDiff, &otherParticle, &otherParticleDiff, true,
        &collisionData, &collisionSolverData, otherNodeIndex, particleIndex, collisionData.gradientMagnitude, &collisionCount, stablizationPass, solverType, particlesPredictedNew, particleCollisionData);
    }
#endif
  GRID_SOLVER_NEIGHBOUR_LOOP_END
#ifdef GRID_COLLISION_SOLVER_USE_SHARED_MEMORY
  localMemBarrier();
  collisionSharedData = localCollisionSharedData[localIndex];
#endif
#else
  uchar validNeighbourIndex[27];
  uchar validNeighbourCount = 0;

  GRID_SOLVER_NEIGHBOUR_LOOP_BEGIN
    const uint2 indexRange = getRangeFromOffset(gridCellParticleOffsets, gridCellIndex);

    validNeighbourIndex[validNeighbourCount++] = encodeCellOffset(i, j, k);
  GRID_SOLVER_NEIGHBOUR_LOOP_END

  for (uchar i=0; i<validNeighbourCount; i++)
  {
#ifndef GRID_SOLVER_HASH_FUNCTION
    const int gridCellIndex = decodeCellIndex(validNeighbourIndex[i], particleGridCellIndex, gridSizeExp);
#else
    const int3 quantizedPosition = positionHashFunction(particleCellPosition + constructFloat3(decodeCellVector(validNeighbourIndex[i]) - constructShort3(1)), gridSize, gridSizeExp);
    const int gridCellIndex = gridIndexInt3Int(quantizedPosition, gridSizeExp);
#endif
    const uint2 indexRange = getRangeFromOffset(gridCellParticleOffsets, gridCellIndex);

    // batchwise iterate over indices in the cell
    for (int otherIndex = indexRange.x; otherIndex < indexRange.y; otherIndex++)
    {
      // iterate over each particle in the loaded batch
      const int otherNodeIndex = gridCellParticleIndices[otherIndex];

      otherParticle = particlesBufferOld[otherNodeIndex];
      if (!shouldCheckForCollision(solverType, particleIndex, otherNodeIndex, &selfParticle, &otherParticle))
        continue;

      const ParticleDifferential otherParticleDiff = particlesDiff[otherNodeIndex];
      positionDiff += processParticleCollision(&selfParticle, &selfParticleDiff, &otherParticle, &otherParticleDiff, true,
        &collisionData, &collisionSolverData, otherNodeIndex, particleIndex, collisionData.gradientMagnitude, &collisionCount, stablizationPass, solverType, particlesPredictedNew, particleCollisionData);
    }
  }

#endif
  positionDiff *= collisionSolverData.collisionDamping;

#ifdef MARK_COLLIDED_PARTICLES
  particleCollisionData[particleIndex].radius = fabs(collisionData.radius) * (collisionCount ? -1.f : 1.f);
#endif

  // apply boundary
  positionDiff += boundaryCollision(&selfParticle, &selfParticleDiff, &collisionData, systemSettings, stablizationPass, &collisionCount, &particleCollisionData[particleIndex], &collisionSolverData);

#ifndef GRID_COLLISION_SOLVE_PAIR_ONCE
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
#else
  atomicAddFloat3(&particlesPredictedNew[particleIndex].position, positionDiff);
  atomicAdd(&particlesPredictedNew[particleIndex].identity.identity, collisionCount);
#endif

#else

  ushort particleSubIndex;
  decodeScatterCellIndex(&particleIndex, &particleSubIndex, particleIndex);

  if (particleSubIndex != 0)
  {
    return;
  }

  const uint gridCellIndex = gridParticleCellIndex[particleIndex * 8];
  short collisionCount = 0;

  // current particle data
  float3 positionDiff = constructFloat3(0.f);
  ParticleStruct selfParticle = particlesBufferOld[particleIndex];
  const IdentityInfo identity = selfParticle.identity;
  const ParticleDifferential selfParticleDiff = particlesDiff[particleIndex];

  ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(identity);
  const PhySystemOffsets phySystemOffsets = systemSettings->globalOffsets[nodeIdentity.solverType];

  nodeIdentity.entityId += phySystemOffsets.globalSolverOffset;
  nodeIdentity.instanceId += phySystemOffsets.globalInstanceOffset;

#ifdef GRID_COLLISION_SOLVER_USE_SHARED_MEMORY
  localCollisionSolverData[localIndex] = particleSharedData[nodeIdentity.entityId].collisionSolverData;
#else
  const CollisionSolverData collisionSolverData = particleSharedData[nodeIdentity.entityId].collisionSolverData;
#endif

  const ParticleCollisionData collisionData = particleCollisionData[particleIndex];

  const ushort solverType = getSolverType(identity);
  const uint2 indexRange = getRangeFromOffset(gridCellParticleOffsets, gridCellIndex);

#ifndef GRID_SOLVER_HASH_FUNCTION
  const float3 inverseMergedBoxSize = ((float)gridSize) / max(gridSize / invRadius[0], systemBoundingBox->max - systemBoundingBox->min);
#endif

  // batchwise iterate over indices in the cell
  for (int otherIndex = indexRange.x; otherIndex < indexRange.y; otherIndex++)
  {
    // iterate over each particle in the loaded batch
    uint otherNodeIndex = gridCellParticleIndices[otherIndex];
    ushort otherNodeSubIndex;

    decodeScatterCellIndex(&otherNodeIndex, &otherNodeSubIndex, otherNodeIndex);

    const ParticleStruct otherParticle = particlesBufferOld[otherNodeIndex];
    if (!shouldCheckForCollision(solverType, particleIndex, otherNodeIndex, &selfParticle, &otherParticle))
      continue;

    const ParticleDifferential otherParticleDiff = particlesDiff[otherNodeIndex];
    positionDiff += processParticleCollision(&selfParticle, &selfParticleDiff, &otherParticle, &otherParticleDiff, otherNodeSubIndex != 0, &collisionData,
#ifdef GRID_COLLISION_SOLVER_USE_SHARED_MEMORY
      &localCollisionSolverData[localIndex],
#else
      &collisionSolverData,
#endif
      otherNodeIndex, particleIndex, collisionData.gradientMagnitude, &collisionCount, stablizationPass, solverType, particlesPredictedNew, particleCollisionData);
  }

#ifdef GRID_COLLISION_SOLVER_USE_SHARED_MEMORY
  positionDiff *= localCollisionSolverData[localIndex].collisionDamping;
#else
  positionDiff *= collisionSolverData.collisionDamping;
#endif

#ifdef MARK_COLLIDED_PARTICLES
  particleCollisionData[particleIndex].radius = fabs(collisionData.radius) * (collisionCount ? -1.f : 1.f);
#endif

  // apply boundary
  positionDiff += boundaryCollision(&selfParticle, &selfParticleDiff, &collisionData, systemSettings, stablizationPass, &collisionCount, &particleCollisionData[particleIndex],
#ifdef GRID_COLLISION_SOLVER_USE_SHARED_MEMORY
    &localCollisionSolverData[localIndex]);
#else
    &collisionSolverData);
#endif

  atomicAddFloat3(&particlesPredictedNew[particleIndex].position, positionDiff);
  atomicAdd(&particlesPredictedNew[particleIndex].identity.identity, collisionCount);

#endif
}

/*
@kernel Apply particle position delta.
@param particlesPredicted Updated particle positions post collision processing.
@param particles Integrated particle position.
@param particlesDelta Particle position delta.
@param nodeCount Total nodes in the solver.
*/
Kernel void applyDeltas(
  Device ParticleStruct*        particlesPredicted,
  Device ParticleStruct*        particles,
  const Device ParticleStruct*  particlesDelta,
  constantKernelInput(uint,     stablizationPass),
  constantKernelInput(uint,     nodeCount)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint particleIndex = threadIndex();

  if (particleIndex >= nodeCount)
  {
    return;
  }

  ParticleStruct particleDelta = particlesDelta[particleIndex];

  if (particleDelta.identity.identity != 0)
  {
    particleDelta.position /= particleDelta.identity.identity;
  }

  const IdentityInfo identity = particlesPredicted[particleIndex].identity;
  particlesPredicted[particleIndex].position += particleDelta.position;
  particlesPredicted[particleIndex].identity = identity;

  if (stablizationPass)
  {
    particles[particleIndex].position += particleDelta.position;
    particles[particleIndex].identity = identity;
  }
}
#endif
