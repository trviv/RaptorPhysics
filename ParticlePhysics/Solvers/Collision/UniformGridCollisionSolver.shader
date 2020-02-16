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
  for (short k=-1; k<2; k++) \
  { \
    for (short j=-1; j<2; j++) \
    { \
      for (short i=-1; i<2; i++) \
      { \
        const int3 quantizedPosition = positionHashFunction(particleCellPosition + constructFloat3(i, j, k), gridSize, gridSizeExp); \
        const int gridCellIndex = gridIndexInt3Int(quantizedPosition, gridSizeExp);

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
        const int gridCellIndex = gridIndexInt3Int(constructInt3(x, y, z), gridSizeExp);

#endif

#define GRID_SOLVER_NEIGHBOUR_LOOP_END \
      } \
    } \
  }

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
  , sharedMemKernelInput(CollisionSolverData, localCollisionSolverData, 16)
  , sharedMemKernelInput(ParticleCollisionData, localCollisionData, 17)
  , sharedMemKernelInput(ParticleStruct, localSelfParticle, 18)
  , sharedMemKernelInput(ParticleStruct, localOtherParticle, 19)
  , sharedMemKernelInput(float3, localPositionDiff, 20)
  , sharedMemKernelInput(ParticleDifferential, localSelfParticleDiff, 21)
  , sharedMemKernelInput(uint3, localTestQueue, 22)
  , sharedMemKernelInput(uint, localCollisionCount, 23)
  , sharedMemKernelInput(uint, localSolverType, 24)
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

#ifdef MARK_COLLIDED_PARTICLES
  bool collided = false;
#endif

  ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(identity);
  const PhySystemOffsets phySystemOffsets = systemSettings->globalOffsets[nodeIdentity.solverType];

  nodeIdentity.entityId += phySystemOffsets.globalSolverOffset;
  nodeIdentity.instanceId += phySystemOffsets.globalInstanceOffset;

#ifdef GRID_COLLISION_SOLVER_USE_SHARED_MEMORY
  localCollisionSolverData[localIndex] = particleSharedData[nodeIdentity.entityId].collisionSolverData;
  localCollisionData[localIndex] = particleCollisionData[particleIndex];
  localSelfParticle[localIndex] = selfParticle;
  localSelfParticleDiff[localIndex] = particlesDiff[particleIndex];
  localPositionDiff[localIndex] = 0.f;
  localCollisionCount[localIndex] = 0;
  localSolverType[localIndex] = getSolverType(identity);

#define collisionSolverData localCollisionSolverData[localIndex]
#define collisionData       localCollisionData[localIndex]
#define selfParticle        localSelfParticle[localIndex]
#define selfParticleDiff    localSelfParticleDiff[localIndex]
#define positionDiff        localPositionDiff[localIndex]
#define otherParticle       localOtherParticle[localIndex]
#define collisionCount      localCollisionCount[localIndex]
#define solverType          localSolverType[localIndex]

  Shared int testCount;
  Shared int allThreadsFinished;
  if (localIndex == 0)
  {
    testCount = 0;
    allThreadsFinished = 0;
  }

  atomicAddShared(&allThreadsFinished, 1);
  localMemBarrier();

  // TODO: revisit logic to find out how to calculate this
  const short workgroupThreads = allThreadsFinished;
  //min(nodeCount - (particleIndex/ComputeSimdWidth)*ComputeSimdWidth, (uint)ComputeSimdWidth);
#else
  const CollisionSolverData collisionSolverData = particleSharedData[nodeIdentity.entityId].collisionSolverData;
  const ParticleCollisionData collisionData = particleCollisionData[particleIndex];
  const ParticleDifferential selfParticleDiff = particlesDiff[particleIndex];
  float3 positionDiff = constructFloat3(0.f);
  uint collisionCount = 0;
  const ushort solverType = getSolverType(identity);
  ParticleStruct otherParticle;
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
    if (localIndex == 0)
    {
      allThreadsFinished = 0;
    }
    localMemBarrier();

    const uint2 indexRange = getRangeFromOffset(gridCellParticleOffsets, gridCellIndex);
    bool cellDone = (indexRange.x == indexRange.y);
    if (cellDone)
    {
      atomicAddShared(&allThreadsFinished, 1);
    }
    localMemBarrier();

    // batchwise iterate over indices in the cell
    for (int otherIndex = indexRange.x; allThreadsFinished < workgroupThreads | testCount > 0; )
    {
      const short threadTestCount = testCount;
      if (threadTestCount >= workgroupThreads | (allThreadsFinished == workgroupThreads & localIndex < threadTestCount))
      {
        const uint3 localTestData = localTestQueue[localIndex];
        const ParticleDifferential otherParticleDiff = particlesDiff[localTestData.y];
        uint collisions = 0;

        const ParticleCollisionData otherCollisionData = localCollisionData[localTestData.z];

        const float3 diff = processParticleCollision(&localSelfParticle[localTestData.z], &localSelfParticleDiff[localTestData.z], &localOtherParticle[localTestData.z], &otherParticleDiff, true, &otherCollisionData, &localCollisionSolverData[localTestData.z], localTestData.y, localTestData.x, otherCollisionData.gradientMagnitude, &collisions, stablizationPass, localSolverType[localTestData.z], particlesPredictedNew,
#ifdef MARK_COLLIDED_PARTICLES
          particleCollisionData, &collided);
#else
          particleCollisionData);
#endif

        atomicAddFloat3Shared(&localPositionDiff[localTestData.z], diff);
        atomicAddShared(&localCollisionCount[localTestData.z], collisions);
        localMemBarrier();

        if (threadTestCount >= workgroupThreads)
        {
          localTestQueue[localIndex] = localTestQueue[workgroupThreads + localIndex];
        }
        if (localIndex == 0)
        {
          testCount = select(0, testCount - workgroupThreads, threadTestCount >= workgroupThreads);
        }
      }

      localMemBarrier();

      if (!cellDone)
      {
        // iterate over each particle in the loaded batch
        const int otherNodeIndex = gridCellParticleIndices[otherIndex];
        otherParticle = particlesBufferOld[otherNodeIndex];
        if (++otherIndex == indexRange.y)
        {
          cellDone = true;
          atomicAddShared(&allThreadsFinished, 1);
        }

        // basic check to determine if collision can happen between the particles
        if (shouldCheckForCollision(getSolverType(selfParticle.identity), particleIndex, otherNodeIndex, &selfParticle, &otherParticle))
        {
          // add to test list at an available offset
          const ushort testQueueIndex = atomicAddSignedShared(&testCount, 1);
          // store test info
          localTestQueue[testQueueIndex] = uint3(particleIndex, otherNodeIndex, localIndex);
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

      otherParticle = particlesBufferOld[otherNodeIndex];
      if (!shouldCheckForCollision(solverType, particleIndex, otherNodeIndex, &selfParticle, &otherParticle))
        continue;

      const ParticleDifferential otherParticleDiff = particlesDiff[otherNodeIndex];
      positionDiff += processParticleCollision(&selfParticle, &selfParticleDiff, &otherParticle, &otherParticleDiff, true,
        &collisionData, &collisionSolverData, otherNodeIndex, particleIndex, collisionData.gradientMagnitude, &collisionCount, stablizationPass, solverType, particlesPredictedNew,
#ifdef MARK_COLLIDED_PARTICLES
        particleCollisionData, &collided);
#else
        particleCollisionData);
#endif
    }
#endif
  GRID_SOLVER_NEIGHBOUR_LOOP_END
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
        &collisionData, &collisionSolverData, otherNodeIndex, particleIndex, collisionData.gradientMagnitude, &collisionCount, stablizationPass, solverType, particlesPredictedNew,
#ifdef MARK_COLLIDED_PARTICLES
        particleCollisionData, &collided);
#else
        particleCollisionData);
#endif
    }
  }

#endif
  positionDiff *= collisionSolverData.collisionDamping;

  // apply boundary
  positionDiff += boundaryCollision(&selfParticle, &selfParticleDiff, &collisionData, systemSettings, stablizationPass, &collisionCount,
#ifdef MARK_COLLIDED_PARTICLES
    &particleCollisionData[particleIndex], &collisionSolverData);
#endif
    &collisionSolverData);

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

#ifdef MARK_COLLIDED_PARTICLES
  particleCollisionData[particleIndex].radius = fabs(collisionData.radius) * (collided ? -1.f : 1.f);
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

#ifdef MARK_COLLIDED_PARTICLES
  bool collided = false;
#endif

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
    positionDiff += processParticleCollision(&selfParticle, &selfParticleDiff, &otherParticle, &otherParticleDiff, otherNodeSubIndex != 0,
#ifdef GRID_COLLISION_SOLVER_USE_SHARED_MEMORY
      &collisionData, &localCollisionSolverData[localIndex], otherNodeIndex, particleIndex, collisionData.gradientMagnitude, &collisionCount, stablizationPass, solverType, particlesPredictedNew,
#else
      &collisionData, &collisionSolverData, otherNodeIndex, particleIndex, collisionData.gradientMagnitude, &collisionCount, stablizationPass, solverType, particlesPredictedNew,
#endif
#ifdef MARK_COLLIDED_PARTICLES
      particleCollisionData, &collided);
#else
      particleCollisionData);
#endif
  }

#ifdef GRID_COLLISION_SOLVER_USE_SHARED_MEMORY
  positionDiff *= localCollisionSolverData[localIndex].collisionDamping;
#else
  positionDiff *= collisionSolverData.collisionDamping;
#endif

  // apply boundary
  positionDiff += boundaryCollision(&selfParticle, &selfParticleDiff, &collisionData, systemSettings, stablizationPass, &collisionCount,
#ifdef MARK_COLLIDED_PARTICLES
    &particleCollisionData[particleIndex],
#endif
#ifdef GRID_COLLISION_SOLVER_USE_SHARED_MEMORY
    &localCollisionSolverData[localIndex]);
#else
    &collisionSolverData);
#endif

  atomicAddFloat3(&particlesPredictedNew[particleIndex].position, positionDiff);
  atomicAdd(&particlesPredictedNew[particleIndex].identity.identity, collisionCount);

#ifdef MARK_COLLIDED_PARTICLES
  particleCollisionData[particleIndex].radius = fabs(collisionData.radius) * (collided ? -1.f : 1.f);
#endif

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
