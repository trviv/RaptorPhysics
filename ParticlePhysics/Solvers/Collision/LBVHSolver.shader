#ifndef LBVH_SOLVER_SHADER
#define LBVH_SOLVER_SHADER

/*
@kernel Compute and store bounding boxes for each particle.
@param particleBoundingBoxes Particle bounding box array.
@param particlesPredicted Integrated particle position.
@param particleSharedData Particle entity shared data.
@param particleAuxData Additional particle data.
@param partitions Instance partition data.
@param entityLocation Entity section data.
@param globalOffsets Offsets to particle nodes all the solvers.
@param nodeCount Total nodes in the solver.
*/
Kernel void createBoundingBoxes(
  Device XAB*                       particleBoundingBoxes,
  Device XAB*                       particleGroupBoundingBoxes,
  const Device ParticleStruct*      particlesPredicted,
  const Device ParticleSharedData*  particleSharedData,
  const Device ParticleAuxData*     particleAuxData,
  const Device PartitionInfo*       partitions,
  const Device EntityLocation*      entityLocation,
  Const PhySystemOffsets*           globalOffsets,
  const uint                        nodeBatchSize,
  const uint                        nodeCount)
{
  // bounding box for the batch
  XAB accumulatedBoundingBox;

  // reset to INF, -INF
  clearXAB(&accumulatedBoundingBox, 0);

  for (uint index = threadIndex(); index < nodeCount; index += threadGroupCount() * threadGroupSize())
  {
    const ParticleStruct particle = particlesPredicted[index];

    ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(particle.identity);

    const PhySystemOffsets phySystemOffsets = globalOffsets[nodeIdentity.solverType];

    nodeIdentity.entityId += phySystemOffsets.globalSolverOffset;
    nodeIdentity.instanceId += phySystemOffsets.globalInstanceOffset;

    const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];
    const ParticleNodeLocator nodeLocator = getNodeLocator(index, phySystemOffsets.globalNodeOffset + partitions[nodeIdentity.instanceId].offset, entityLocation[nodeIdentity.entityId].node);

    const float radius = getRadiusUsingDeviceAux(&sharedData, particleAuxData, nodeLocator.commonNodeIndex);

    XAB particleBoundingBox;
    particleBoundingBox.min = particle.position - constructFloat3(radius);
    particleBoundingBox.max = particle.position + constructFloat3(radius);
    particleBoundingBoxes[index] = particleBoundingBox;

    mergeXAB(&accumulatedBoundingBox, &particleBoundingBox);
  }

  if (threadIndex() < nodeBatchSize)
  {
    particleGroupBoundingBoxes[threadIndex()] = accumulatedBoundingBox;
  }
}

/*
@kernel Compute and store morton code for each particle.
@param bvhLeafs Particle position and index data for bounding volume hierarchy.
@param mergedParticleBoundingBox Bounds for the complete scene.
@param particles Particle positions.
@param nodeCount Total nodes in the solver.
*/
Kernel void assignMortonCode(
  Device BVHLeafInfo*           bvhLeafs,
  Const XAB*                    mergedParticleBoundingBox,
  const Device ParticleStruct*  particles,
  const uint                    nodeCount)
{
  const uint index = threadIndex();

  if (index < nodeCount)
  {
    const XAB mergedBox = mergedParticleBoundingBox[0];
    const float3 inverseMergedBoxSize = ((float)1024) / (mergedBox.max - mergedBox.min);
    const float3 mergedBoxCenter = (mergedBox.min + mergedBox.max) * 0.5f;

    const ParticleStruct particle = particles[index];

    // Quantize into integer coordinates
    // floor() is needed to prevent the center cell, at (0,0,0) from being twice the size
    float3 positionRelativeToCenter = (particle.position - mergedBoxCenter) * inverseMergedBoxSize;

    int3 quantizedPosition = convert_int3(select(floor(positionRelativeToCenter), positionRelativeToCenter, positionRelativeToCenter >= 0.0f));

    // Clamp coordinates into [-512, 511], then convert range from [-512, 511] to [0, 1023]
    quantizedPosition = max(constructInt3(-512), min(quantizedPosition, constructInt3(511))) + constructInt3(512);

    //Interleave bits(assign a morton code, also known as a z-curve)
    BVHLeafInfo bvhLeaf;
    bvhLeaf.mortonCode = get32BitMortonCode(quantizedPosition);
    bvhLeaf.index = index;

    bvhLeafs[index] = bvhLeaf;
  }
}

//#define DEBUG_TREE_CREATION

//The most significant bit(0x80000000) of a int32 is used to distinguish between leaf and internal nodes.
//If it is set, then the index is for an internal node; otherwise, it is a leaf node. 
//In both cases, the bit should be cleared to access the actual node index.
int isLeafNode(int index)
{
#ifdef DEBUG_TREE_CREATION
  return (index - 1000000000) < 0;
#else
  return (index & 0x80000000) == 0;
#endif
}

int setInternalNodeMarker(int isLeaf, int index)
{
#ifdef DEBUG_TREE_CREATION
  return select(index + 1000000000, index, isLeaf);
#else
  return select((int)(index | 0x80000000), index, isLeaf);
#endif
}

int removeInternalNodeMarker(int index)
{
#ifdef DEBUG_TREE_CREATION
  return index - 1000000000;
#else
  return index & (~0x80000000);
#endif
}

int getCommonPrefixLength(const int2 left, const int2 right)
{
  int ret = clz(left.x ^ right.x);
  return select(ret, clz(left.y ^ right.y) + 32, ret == 32);
}

#define LBVH_ROOT_NODE_MARKER       ((int)-1)

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
  const Device BVHLeafInfo* bvhLeafs,
  const int                 nodeCount)
{
  const uint internalNodeCount = nodeCount - 1;
  const int internalNodeIndex = threadIndex();

  if (internalNodeIndex < internalNodeCount)
  {
    //#define USE_LINEAR_SEARCH
#ifdef USE_LINEAR_SEARCH
//#define LBVH_INVALID_COMMON_PREFIX  ((int)128)
//
//    const uint2 nodePrefix = leafNodeCommonPrefixes[internalNodeIndex];
//    const int nodePrefixLength = leafNodeCommonPrefixLengths[internalNodeIndex];
//
//    int leftIndex = -1;
//    int rightIndex = -1;
//
//    // Find nearest element to left with a lower common prefix
//    for (int i = internalNodeIndex - 1; i >= 0; i--)
//    {
//      int nodeLeftSharedPrefixLength = getSharedPrefixLength(nodePrefix, leafNodeCommonPrefixes[i], nodePrefixLength, leafNodeCommonPrefixLengths[i]);
//      if (nodeLeftSharedPrefixLength < nodePrefixLength)
//      {
//        leftIndex = i;
//        break;
//      }
//    }
//
//    // Find nearest element to right with a lower common prefix
//    for (int i = internalNodeIndex + 1; i < internalNodeCount; i++)
//    {
//      int nodeRightSharedPrefixLength = getSharedPrefixLength(nodePrefix, leafNodeCommonPrefixes[i], nodePrefixLength, leafNodeCommonPrefixLengths[i]);
//      if (nodeRightSharedPrefixLength < nodePrefixLength)
//      {
//        rightIndex = i;
//        break;
//      }
//    }
//
//    // Select parent
//    {
//      const int leftPrefixLength = (leftIndex != -1) ? leafNodeCommonPrefixLengths[leftIndex] : LBVH_INVALID_COMMON_PREFIX;
//      const int rightPrefixLength = (rightIndex != -1) ? leafNodeCommonPrefixLengths[rightIndex] : LBVH_INVALID_COMMON_PREFIX;
//
//      uint isLeftHigherPrefixLength = (leftPrefixLength > rightPrefixLength);
//
//      if (leftPrefixLength == LBVH_INVALID_COMMON_PREFIX)       isLeftHigherPrefixLength = 0;
//      else if (rightPrefixLength == LBVH_INVALID_COMMON_PREFIX) isLeftHigherPrefixLength = 1;
//
//      const int parentNodeIndex = (isLeftHigherPrefixLength) ? leftIndex : rightIndex;
//
//      const uint isRootNode = (leftIndex == -1 && rightIndex == -1);
//      internalNodeParentIndices[internalNodeIndex] = (!isRootNode) ? parentNodeIndex : LBVH_ROOT_NODE_MARKER;
//
//      if (!isRootNode)
//      {
//        int isRightChild = (isLeftHigherPrefixLength);  //If the left node is the parent, then this node is its right child and vice versa
//
//        Device int* childNodesAsInt = (Device int*)&internalNodeChildIndices[parentNodeIndex];
//        childNodesAsInt[isRightChild] = setInternalNodeMarker(0, internalNodeIndex);
//      }
//      else
//      {
//        out_rootNodeIndex[0] = setInternalNodeMarker(0, internalNodeIndex);
//      }
//    }
//
#else

    // code of leaf before this
    int2 prevNodePrefix;
    if (internalNodeIndex > 0)
    {
      prevNodePrefix = constructInt2(bvhLeafs[internalNodeIndex - 1].mortonCode, internalNodeIndex - 1);
    }
    // code of this leaf
    const int2 currNodePrefix = constructInt2(bvhLeafs[internalNodeIndex].mortonCode, internalNodeIndex);
    // code of leaf after this
    const int2 nextNodePrefix = constructInt2(bvhLeafs[internalNodeIndex + 1].mortonCode, internalNodeIndex + 1);

    const int nodeLeftSharedPrefixLength = select(-1, getCommonPrefixLength(currNodePrefix, prevNodePrefix), internalNodeIndex > 0);
    const int nodeRightSharedPrefixLength = getCommonPrefixLength(currNodePrefix, nextNodePrefix);

    const int direction = select(-1, 1, nodeRightSharedPrefixLength > nodeLeftSharedPrefixLength);
    const int minLength = select(nodeRightSharedPrefixLength, nodeLeftSharedPrefixLength, nodeRightSharedPrefixLength > nodeLeftSharedPrefixLength);

    int maxStep = 128;
    int nextInternalNodeIndex = internalNodeIndex + maxStep * direction;

    // Compute upper bound for the length of the range
    while (nextInternalNodeIndex >= 0 && nextInternalNodeIndex < nodeCount &&
      getCommonPrefixLength(currNodePrefix, constructInt2(bvhLeafs[nextInternalNodeIndex].mortonCode, nextInternalNodeIndex)) > minLength)
    {
      maxStep *= 4;
      nextInternalNodeIndex = internalNodeIndex + maxStep * direction;
    }

    int endPosition = 0;
    int currentStep = maxStep >> 1;

    // Find the other end using binary search
    while (currentStep != 0)
    {
      nextInternalNodeIndex = internalNodeIndex + (currentStep + endPosition) * direction;
      if (nextInternalNodeIndex >= 0 && nextInternalNodeIndex < nodeCount &&
        getCommonPrefixLength(currNodePrefix, constructInt2(bvhLeafs[nextInternalNodeIndex].mortonCode, nextInternalNodeIndex)) > minLength)
      {
        endPosition += currentStep;
      }
      currentStep >>= 1;
    }

    int otherInternalNodeIndex = internalNodeIndex + endPosition * direction;
    const int splitLength = getCommonPrefixLength(currNodePrefix, constructInt2(bvhLeafs[otherInternalNodeIndex].mortonCode, otherInternalNodeIndex));

    int splitPosition = 0;

    // Find the split position using binary search
    do
    {
      endPosition = (endPosition >> 1) + (endPosition & 1);
      nextInternalNodeIndex = internalNodeIndex + (endPosition + splitPosition) * direction;
      if (nextInternalNodeIndex >= 0 && nextInternalNodeIndex < nodeCount &&
        getCommonPrefixLength(currNodePrefix, constructInt2(bvhLeafs[nextInternalNodeIndex].mortonCode, nextInternalNodeIndex)) > splitLength)
      {
        splitPosition += endPosition;
      }
    } while (endPosition > 1);

    splitPosition = internalNodeIndex + splitPosition * direction + min(direction, 0);

    int leftIsLeaf = (min(internalNodeIndex, otherInternalNodeIndex) == splitPosition);
    int rightIsLeaf = (max(internalNodeIndex, otherInternalNodeIndex) == (splitPosition + 1));

    treeInternalNodes[internalNodeIndex].child[0] = setInternalNodeMarker(leftIsLeaf, splitPosition);
    treeInternalNodes[internalNodeIndex].child[1] = setInternalNodeMarker(rightIsLeaf, splitPosition + 1);

    // mark internal node at zero'th index as the root
    if (internalNodeIndex == 0)
    {
      treeInternalNodes[internalNodeIndex].parent = LBVH_ROOT_NODE_MARKER;
    }

    if (!leftIsLeaf)
    {
      treeInternalNodes[splitPosition].parent = internalNodeIndex;
    }
    if (!rightIsLeaf)
    {
      treeInternalNodes[splitPosition + 1].parent = internalNodeIndex;
    }

    if (leftIsLeaf)
    {
      leafParentNodeIndices[splitPosition] = internalNodeIndex;
    }
    if (rightIsLeaf)
    {
      leafParentNodeIndices[splitPosition + 1] = internalNodeIndex;
    }

    // reset visited counter for use in next dispatch
    visitedInternalNodes[internalNodeIndex] = 0;
#endif
  }
}


/*
@kernel Create boundig box of tree nodes.
@param treeInternalNodeBoundingBoxes Bounding box for tree nodes.
@param visitedInternalNodes Counter for tracking threads visited during bottom up traversal.
@param treeInternalNodes Binary radix tree internal connectivity data.
@param leafParentNodeIndices Immediate parents to particle leaf data.
@param particleBoundingBoxes Particle bounding box array.
@param nodeCount Total nodes in the solver.
*/
Kernel void constructTreeBoundingBox(
  Device XAB*               treeInternalNodeBoundingBoxes,
  Device uint*              visitedInternalNodes,
  const Device BVHNodeInfo* treeInternalNodes,
  const Device uint*        leafParentNodeIndices,
  const Device XAB*         particleBoundingBoxes,
  const int                 nodeCount)
{
  int index = threadIndex();

  if (index < nodeCount)
  {
    // process leaf node bounding boxes first
    uint currentNodeIndex = leafParentNodeIndices[index];
    // get the processing order
    uint visited = atomicAdd((visitedInternalNodes + currentNodeIndex), 1);
    // get internal node for the leaf
    BVHNodeInfo internalNode = treeInternalNodes[currentNodeIndex];

    // only process if leaf
    while (visited)
    {
      // bounding box accumulated by the thread
      XAB mergedBoundingBox;
      clearXAB(&mergedBoundingBox, 0);

      // fetched bounding box
      XAB childBoundingBox;
      if (isLeafNode(internalNode.child[0]))
      {
        childBoundingBox = particleBoundingBoxes[internalNode.child[0]];
      }
      else
      {
        childBoundingBox = treeInternalNodeBoundingBoxes[removeInternalNodeMarker(internalNode.child[0])];
      }
      mergeXAB(&mergedBoundingBox, &childBoundingBox);

      if (isLeafNode(internalNode.child[1]))
      {
        childBoundingBox = particleBoundingBoxes[internalNode.child[1]];
      }
      else
      {
        childBoundingBox = treeInternalNodeBoundingBoxes[removeInternalNodeMarker(internalNode.child[1])];
      }
      mergeXAB(&mergedBoundingBox, &childBoundingBox);

      // save the internal node information
      treeInternalNodeBoundingBoxes[currentNodeIndex] = mergedBoundingBox;

      // process the parent node next
      currentNodeIndex = internalNode.parent;

      // exit if root
      if (currentNodeIndex == LBVH_ROOT_NODE_MARKER)
      {
        break;
      }
      // fetch the node data
      internalNode = treeInternalNodes[currentNodeIndex];
      // get the visited order
      visited = atomicAdd((visitedInternalNodes + currentNodeIndex), 1);
    }
  }
}


int intersectXAB(const Thread XAB* a, const Thread XAB* b)
{
  const int3 ret = (a->min < b->max) && (a->max > b->min);
  return (ret.x && ret.y && ret.z);
  //return (a->min.x < b->max.x && a->max.x > b->min.x)
  //  && (a->min.y < b->max.y && a->max.y > b->min.y)
  //  && (a->min.z < b->max.z && a->max.z > b->min.z);
}

//#define MARK_COLLIDED_PARTICLES
//#define DEBUG_TRAVERSAL

inline ParticleStruct traverseBinaryTree(
  const Device ParticleStruct*        particlesPredictedOld,
  const Device BVHNodeInfo*           treeInternalNodes,
  const Device XAB*                   treeInternalNodeBoundingBoxes,
  const Device XAB*                   particleBoundingBoxes,
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
  const int                           index)
{
  ParticleStruct output;

  uint* stackTop;
  uint  traversalStack[64];

  stackTop = &traversalStack[0];

#ifdef MARK_COLLIDED_PARTICLES
  bool collided = false;
#endif

  const ParticleStruct predicted = particlesPredictedOld[index];
  output = predicted;
  //ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(predicted.identity);
  //const PhySystemOffsets phySystemOffsets = globalOffsets[nodeIdentity.solverType];

  //nodeIdentity.entityId += phySystemOffsets.globalSolverOffset;
  //nodeIdentity.instanceId += phySystemOffsets.globalInstanceOffset;

  //const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];
  //const ParticleCollisionData collisionData = getSDFUsingDeviceCollision(&sharedData, particleCollisionData, index);
  const ParticleCollisionData collisionData = particleCollisionData[index];

  XAB particleBoundingBox;
  particleBoundingBox.min = predicted.position - constructFloat3(collisionData.radius);
  particleBoundingBox.max = predicted.position + constructFloat3(collisionData.radius);

  const float sdfMagnitude = length(collisionData.transformedSdfGradient);

  // mark index of the root node internal
  uint currentNodeIndex = setInternalNodeMarker(0, 0);

  while (true)
  {
#ifdef DEBUG_TRAVERSAL
    printf("Node: %d %d %d\n", index, currentNodeIndex, isLeafNode(currentNodeIndex));
#endif

    // traverse while a leaf node is found
    while (!isLeafNode(currentNodeIndex))
    {
      const BVHNodeInfo node = treeInternalNodes[removeInternalNodeMarker(currentNodeIndex)];

      const XAB leftBoundingBox = treeInternalNodeBoundingBoxes[select(removeInternalNodeMarker(node.child[0]), (int)node.child[0], isLeafNode(node.child[0]))];
      const XAB rightBoundingBox = treeInternalNodeBoundingBoxes[select(removeInternalNodeMarker(node.child[1]), (int)node.child[1], isLeafNode(node.child[1]))];

      int isIntersectingLeft = intersectXAB(&particleBoundingBox, &leftBoundingBox);
      int isIntersectingRight = intersectXAB(&particleBoundingBox, &rightBoundingBox);

      if (isIntersectingLeft)
      {
        *(stackTop++) = node.child[0];
#ifdef DEBUG_TRAVERSAL
        printf("Stack Push: %d %d\n", index, node.child[0]);
#endif
      }
      if (isIntersectingRight)
      {
        *(stackTop++) = node.child[1];
#ifdef DEBUG_TRAVERSAL
        printf("Stack Push: %d %d\n", index, node.child[1]);
#endif
      }

      // break if the stack is empty
      if (stackTop == traversalStack)
      {
        // mark node invalid
        currentNodeIndex = LBVH_ROOT_NODE_MARKER;
        break;
      }

      currentNodeIndex = *(--stackTop);
#ifdef DEBUG_TRAVERSAL
      printf("Stack Pop: %d %d\n", index, currentNodeIndex);
#endif
    }

#ifdef DEBUG_TRAVERSAL
    printf("Test: %d %d\n", index, currentNodeIndex);
#endif
    // while ()
    // test colision if not an invalid node
    if (currentNodeIndex != LBVH_ROOT_NODE_MARKER)
    {
      const ParticleStruct predicted2 = particlesPredictedOld[currentNodeIndex];

      if (predicted2.identity.identity != predicted.identity.identity)
      {
        //ParticleNodeIdentity nodeIdentity2 = uncompressToNodeIdentity(predicted2.identity);
        //const PhySystemOffsets phySystemOffsets2 = globalOffsets[nodeIdentity2.solverType];

        //nodeIdentity2.entityId += phySystemOffsets.globalSolverOffset;
        //nodeIdentity2.instanceId += phySystemOffsets.globalInstanceOffset;

        //const ParticleSharedData sharedData2 = particleSharedData[nodeIdentity2.entityId];
        //const ParticleCollisionData collisionData2 = getSDFUsingDeviceCollision(&sharedData, particleCollisionData, currentNodeIndex);

        const ParticleCollisionData collisionData2 = particleCollisionData[currentNodeIndex];

        // skip if the base and the batch particle are of the same object
        const float3 distanceVector = predicted.position - predicted2.position;
        const float actualDistance = dot(distanceVector, distanceVector);

#ifdef MARK_COLLIDED_PARTICLES
        const float allowedDistance = sqr(fabs(collisionData2.radius) + fabs(collisionData.radius));
#else
        const float allowedDistance = sqr(collisionData2.radius + collisionData.radius);
#endif
        // if overlapping
        if (actualDistance < allowedDistance)
        {
          const float sdfMagnitude2 = length(collisionData2.transformedSdfGradient);

          float3 normal = select(-collisionData2.transformedSdfGradient, collisionData.transformedSdfGradient, constructUint3(sdfMagnitude < sdfMagnitude2));
          float3 delta = normal * (collisionData.invMass / (collisionData.invMass + collisionData2.invMass));
          output.position -= delta;
#ifdef MARK_COLLIDED_PARTICLES
          collided = true;
#endif
        }
      }
    }

    // exit if nothing to fetch
    if (stackTop == traversalStack)
    {
      break;
    }

    // pop from the stack
    currentNodeIndex = *(--stackTop);
  }
  return output;
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
  volatile Device uint*               batchCounter,
  Device ParticleStruct*              particlesPredictedNew,
  const Device ParticleStruct*        particlesPredictedOld,
  const Device BVHNodeInfo*           treeInternalNodes,
  const Device XAB*                   treeInternalNodeBoundingBoxes,
  const Device XAB*                   particleBoundingBoxes,
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
  const uint                          nodeCount)
{
#define batchMultiple 2

  volatile Shared int batchOffset[32];
  volatile Shared int batchCount[32];

  if (threadIndex() == 0)
  {
    atomicStore(batchCounter, 0);
  }

  localMemBarrier();

  const uint subGroupLocalIndex = threadLocalIndex() & (COMPUTE_SUB_GROUP_SIZE - 1);
  const uint subGroupIndex = threadLocalIndex() >> COMPUTE_SUB_GROUP_EXP;

  if (subGroupLocalIndex == 0)
  {
    batchOffset[subGroupIndex] = 0;
    batchCount[subGroupIndex] = 0;
  }

  uint index;
  while (true)
  {
    if (subGroupLocalIndex == 0 && batchCount[subGroupIndex] == 0)
    {
      batchOffset[subGroupIndex] = atomicAdd(batchCounter, COMPUTE_SUB_GROUP_SIZE * batchMultiple);
      batchCount[subGroupIndex] = COMPUTE_SUB_GROUP_SIZE * batchMultiple;
    }

    index = batchOffset[subGroupIndex] + subGroupLocalIndex;

    if (index >= nodeCount)
    {
      break;
    }

    // within valid grid cell bounds
#ifdef DEBUG_TRAVERSAL
    if (index == 0)
#endif
    {
      particlesPredictedNew[index] = traverseBinaryTree(
        particlesPredictedOld,
        treeInternalNodes,
        treeInternalNodeBoundingBoxes,
        particleBoundingBoxes,
        particleCollisionData,
        particleSharedData,
        particleAuxData,
        partitions,
        entityLocation,
        globalOffsets,
        index);

#ifdef MARK_COLLIDED_PARTICLES
      particleCollisionData[index].radius = fabs(collisionData.radius) * (collided ? -1.f : 1.f);
#endif
    }

    if (subGroupLocalIndex == 0)
    {
      batchOffset[subGroupIndex] += COMPUTE_SUB_GROUP_SIZE;
      batchCount[subGroupIndex] -= COMPUTE_SUB_GROUP_SIZE;
    }
  }
}

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
      }
    }
  }
}

#endif
