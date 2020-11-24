#ifndef BOUNDING_VOLUME_HIERARCHY_ADS_CREATE_SHADER_H
#define BOUNDING_VOLUME_HIERARCHY_ADS_CREATE_SHADER_H

//#define BOUNDING_VOLUME_HIERARCHY_ADS_DEBUG_TREE_CREATION

#define BOUNDING_VOLUME_HIERARCHY_ADS_ROOT_NODE_MARKER  ((uint)-1)

// The most significant bit(0x80000000) of a uint32 is used to distinguish between leaf and internal nodes.
// If it is set, then the index is for an internal node; otherwise, it is a leaf node.
// In both cases, the bit should be cleared to access the actual node index.
inline bool isBVHLeafNode(const uint index)
{
#ifdef BOUNDING_VOLUME_HIERARCHY_ADS_DEBUG_TREE_CREATION
  return index != BOUNDING_VOLUME_HIERARCHY_ADS_ROOT_NODE_MARKER && (index - 1000000000) < 0;
#else
  return (index & 0x80000000) == 0;
#endif
}

inline uint setBVHInternalNodeMarker(const bool isLeaf, uint index)
{
#ifdef BOUNDING_VOLUME_HIERARCHY_ADS_DEBUG_TREE_CREATION
  return select(index + 1000000000, index, isLeaf);
#else
  return select(index | 0x80000000, index, isLeaf);
#endif
}

inline uint removeBVHInternalNodeMarker(uint index)
{
#ifdef BOUNDING_VOLUME_HIERARCHY_ADS_DEBUG_TREE_CREATION
  return select(index - 1000000000, index, index < 1000000000);
#else
  return index & (~0x80000000);
#endif
}

inline short getBVHCommonPrefixLength(const uint2 left, const uint2 right)
{
  short ret = clz(left.x ^ right.x);
  return select(ret, (short)(clz(left.y ^ right.y) + 32), ret == 32);
}


/*
@kernel Create tree from the leaf data.
@param treeInternalNodes Binary radix tree internal connectivity data.
@param visitedInternalNodes Counter for tracking threads visited during bottom up traversal.
@param leafParentNodeIndices Immediate parents to particle leaf data.
@param bvhLeafs Particle position and index data for bounding volume hierarchy.
@param nodeCount Total nodes in the solver.
*/
Kernel void constructBinaryTree(
  Device BVHNodeInfo*       treeInternalNodes,
  Device uint*              visitedInternalNodes,
  Device uint*              leafParentNodeIndices,
  Device uint*              nodeParentNodeIndices,
  const Device BVHLeafInfo* bvhLeafs,
  constantKernelInput(int,  nodeCount)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const int internalNodeIndex = threadIndex();

  if (internalNodeIndex >= (nodeCount - 1))
    return;

  // code of leaf before this
  const uint2 prevNodePrefix = constructUint2(select(0xFFFFFFFF, bvhLeafs[internalNodeIndex - 1].mortonCode, internalNodeIndex > 0), internalNodeIndex - 1);

  // code of this leaf
  const uint2 currNodePrefix = constructUint2(bvhLeafs[internalNodeIndex].mortonCode, internalNodeIndex);

  // code of leaf after this
  const uint2 nextNodePrefix = constructUint2(bvhLeafs[internalNodeIndex + 1].mortonCode, internalNodeIndex + 1);

  const short nodeLeftSharedPrefixLength  = select((short)-1, (short)getBVHCommonPrefixLength(currNodePrefix, prevNodePrefix), internalNodeIndex > 0);
  const short nodeRightSharedPrefixLength = getBVHCommonPrefixLength(currNodePrefix, nextNodePrefix);

  const int direction   = select(-1, 1, nodeRightSharedPrefixLength > nodeLeftSharedPrefixLength);
  const short minLength = select(nodeRightSharedPrefixLength, nodeLeftSharedPrefixLength, nodeRightSharedPrefixLength > nodeLeftSharedPrefixLength);

  int maxStep = 8;
  int currInternalNodeIndex;

  // Compute upper bound for the length of the range
  do
  {
    maxStep <<= 1;
    currInternalNodeIndex = internalNodeIndex + maxStep * direction;
  }
  while (currInternalNodeIndex >= 0 && currInternalNodeIndex < nodeCount && getBVHCommonPrefixLength(currNodePrefix, constructUint2(bvhLeafs[currInternalNodeIndex].mortonCode, currInternalNodeIndex)) > minLength);

  int indexOffset = 0;

  // Find the exact offset using binary search
  for (int currentStep = maxStep >> 1; currentStep != 0; currentStep >>= 1)
  {
    currInternalNodeIndex = internalNodeIndex + (currentStep + indexOffset) * direction;
    if (currInternalNodeIndex >= 0 && currInternalNodeIndex < nodeCount && getBVHCommonPrefixLength(currNodePrefix, constructUint2(bvhLeafs[currInternalNodeIndex].mortonCode, currInternalNodeIndex)) > minLength)
    {
      indexOffset += currentStep;
    }
  }

  const int otherInternalNodeIndex = internalNodeIndex + indexOffset * direction;
  const short splitLength = getBVHCommonPrefixLength(currNodePrefix, constructUint2(bvhLeafs[otherInternalNodeIndex].mortonCode, otherInternalNodeIndex));

  int splitOffset = 0;

  // Find the split position using binary search
  do
  {
    indexOffset = (indexOffset >> 1) + (indexOffset & 1);
    currInternalNodeIndex = internalNodeIndex + (indexOffset + splitOffset) * direction;
    if (currInternalNodeIndex >= 0 && currInternalNodeIndex < nodeCount && getBVHCommonPrefixLength(currNodePrefix, constructUint2(bvhLeafs[currInternalNodeIndex].mortonCode, currInternalNodeIndex)) > splitLength)
    {
      splitOffset += indexOffset;
    }
  }
  while (indexOffset > 1);

  splitOffset = internalNodeIndex + splitOffset * direction + min(direction, 0);

  const bool leftIsLeaf  = (min(internalNodeIndex, otherInternalNodeIndex) == splitOffset);
  const bool rightIsLeaf = (max(internalNodeIndex, otherInternalNodeIndex) == (splitOffset + 1));

  BVHNodeInfo bvhNode;
  bvhNode.childLeft  = setBVHInternalNodeMarker(leftIsLeaf, splitOffset);
  bvhNode.childRight = setBVHInternalNodeMarker(rightIsLeaf, splitOffset + 1);

  treeInternalNodes[internalNodeIndex] = bvhNode;

  // mark internal node at zero'th index as the root
  if (internalNodeIndex == 0)
  {
    nodeParentNodeIndices[internalNodeIndex] = BOUNDING_VOLUME_HIERARCHY_ADS_ROOT_NODE_MARKER;
  }

  if (!leftIsLeaf)  nodeParentNodeIndices[splitOffset]     = internalNodeIndex;
  if (!rightIsLeaf) nodeParentNodeIndices[splitOffset + 1] = internalNodeIndex;

  if (leftIsLeaf)   leafParentNodeIndices[splitOffset]     = internalNodeIndex;
  if (rightIsLeaf)  leafParentNodeIndices[splitOffset + 1] = internalNodeIndex;

  // reset visited counter for use in next dispatch
  visitedInternalNodes[internalNodeIndex] = 0;
}


/*
@kernel Create boundig box of tree nodes.
@param treeInternalNodeBoundingBoxes Bounding box for tree nodes.
@param visitedInternalNodes Counter for tracking threads visited during bottom up traversal.
@param treeInternalNodes Binary radix tree internal connectivity data.
@param leafParentNodeIndices Immediate parents to particle leaf data.
@param primitiveBoundingBoxes Particle bounding box array.
@param nodeCount Total nodes in the solver.
*/
Kernel void constructTreeBoundingBox(
  Device XAB*               treeInternalNodeBoundingBoxes,
  atomicKernelInput(uint,   visitedInternalNodes),
  const Device BVHNodeInfo* treeInternalNodes,
  const Device uint*        leafParentNodeIndices,
  const Device uint*        nodeParentNodeIndices,
  const Device XAB*         primitiveBoundingBoxes,
  constantKernelInput(int,  nodeCount)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint index = threadIndex();

  if (index >= nodeCount)
    return;

  // begin with leaf node bounding boxes
  uint currentNodeIndex = leafParentNodeIndices[index];

  // exit if root
  while (currentNodeIndex != BOUNDING_VOLUME_HIERARCHY_ADS_ROOT_NODE_MARKER)
  {
    // get the visited order
    const uint visited = atomicAdd(&visitedInternalNodes[currentNodeIndex], 1);

    // only process if visited after
    if (visited == 0)
    {
      break;
    }

    // fetch node data
    const BVHNodeInfo internalNode = treeInternalNodes[currentNodeIndex];

    // bounding box accumulated by the thread
    XAB mergedBoundingBox;
    clearXAB(&mergedBoundingBox, INFINITY);

    // fetched bounding box
    XAB childBoundingBox;
    if (isBVHLeafNode(internalNode.childLeft))
    {
      childBoundingBox = primitiveBoundingBoxes[internalNode.childLeft];
    }
    else
    {
      childBoundingBox = treeInternalNodeBoundingBoxes[removeBVHInternalNodeMarker(internalNode.childLeft)];
    }
    mergeXAB(&mergedBoundingBox, &childBoundingBox);

    if (isBVHLeafNode(internalNode.childRight))
    {
      childBoundingBox = primitiveBoundingBoxes[internalNode.childRight];
    }
    else
    {
      childBoundingBox = treeInternalNodeBoundingBoxes[removeBVHInternalNodeMarker(internalNode.childRight)];
    }
    mergeXAB(&mergedBoundingBox, &childBoundingBox);

    // save the internal node information
    treeInternalNodeBoundingBoxes[currentNodeIndex] = mergedBoundingBox;

    // process the parent node next
    currentNodeIndex = nodeParentNodeIndices[currentNodeIndex];
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
@param systemSettings Settings for the physics system.
@param gridParticleCellIndex Computed cell index for each particle.
@param nodeCount Total nodes in the solver.
@param occupiedCellCount Total active grid cells.
*/
/*Kernel void applyCollisions(
  atomicKernelInput(uint,             batchCounter),
  Device ParticleStruct*              particlesPredictedNew,
  Device ParticleStruct*              particlesInit,
  const Device ParticleDifferential*  particlesDiff,
  const Device ParticleStruct*        particlesPredictedOld,
  const Device BVHNodeInfo*           treeInternalNodes,
  const Device uint*                  leafParentNodeIndices,
  const Device uint*                  nodeParentNodeIndices,
  const Device XAB*                   treeInternalNodeBoundingBoxes,
#ifdef MARK_COLLIDED_PARTICLES
  Device ParticleCollisionData*       particleCollisionData,
#else
  const Device ParticleCollisionData* particleCollisionData,
#endif
  const Device ParticleSharedData*    particleSharedData,
  Const PhySystemSettings*            systemSettings,
  constantKernelInput(uint,           nodeCount),
  constantKernelInput(ushort,         stablizationPass)
  KERNEL_GLOBAL_ARGUMENTS
  KERNEL_THREAD_ARGUMENTS)
{
#define batchMultiple 1

  volatile Shared int batchOffset[32];
  volatile Shared short batchCount[32];

  if (threadIndex() == 0)
  {
    atomicStore(batchCounter, 0);
  }

  localMemBarrier();

  const uchar subGroupLocalIndex = threadLocalIndex() & (ComputeSimdWidth - 1);
  const uchar subGroupIndex = threadLocalIndex() >> ComputeSimdWidthExp;

  if (subGroupLocalIndex == 0)
  {
    batchOffset[subGroupIndex] = 0;
    batchCount[subGroupIndex] = 0;
  }

  INIT_POLL();

  // process until all batches are exhausted
  while (!POLL_TIMEOUT())
  {
    if (subGroupLocalIndex == 0 && batchCount[subGroupIndex] == 0)
    {
      batchOffset[subGroupIndex] = atomicAdd(batchCounter, ComputeSimdWidth * batchMultiple);
      batchCount[subGroupIndex] = ComputeSimdWidth * batchMultiple;
    }

    const uint index = batchOffset[subGroupIndex] + subGroupLocalIndex;

    if (index >= nodeCount)
    {
      break;
    }

    // within valid grid cell bounds
#ifdef DEBUG_TRAVERSAL
    if (index == 0)
#endif
    {
      // get all the data for particle being processed
      ParticleStruct currentParticle = particlesPredictedOld[index];
      const IdentityInfo identity = currentParticle.identity;
      ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(identity);
      const PhySystemOffsets phySystemOffsets = systemSettings->globalOffsets[nodeIdentity.solverType];

      nodeIdentity.entityId += phySystemOffsets.globalSolverOffset;
      nodeIdentity.instanceId += phySystemOffsets.globalInstanceOffset;

      //ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];
      CollisionSolverData collisionSolverData = particleSharedData[nodeIdentity.entityId].collisionSolverData;
      ParticleCollisionData collisionData = particleCollisionData[index];

      const ParticleDifferential selfParticleDiff = particlesDiff[index];
      const ushort solverType = getSolverType(identity);

      // find position change due to collision
      float3 delta = stackTraverseBinaryTree(
        &currentParticle,
        particlesPredictedOld,
        selfParticleDiff,
        particlesDiff,
        treeInternalNodes,
        leafParentNodeIndices,
        nodeParentNodeIndices,
        treeInternalNodeBoundingBoxes,
        &collisionData,
        stablizationPass,
        solverType,
        particlesPredictedNew,
        particleCollisionData,
        &collisionSolverData,
        index);

      delta *= collisionSolverData.collisionDamping;

      uint collisionCount = 0;
      // apply boundary
      delta += boundaryCollision(&currentParticle, &selfParticleDiff, &collisionData, systemSettings, stablizationPass, &collisionCount, &particleCollisionData[index], &collisionSolverData);

      // update position
      particlesPredictedNew[index].position += delta;
      particlesPredictedNew[index].identity = identity;

      if (stablizationPass)
      {
        particlesInit[index].position += delta;
        particlesInit[index].identity = identity;
      }
    }

    if (subGroupLocalIndex == 0)
    {
      batchOffset[subGroupIndex] += ComputeSimdWidth;
      batchCount[subGroupIndex] -= ComputeSimdWidth;
    }
  }
}*/

#endif
