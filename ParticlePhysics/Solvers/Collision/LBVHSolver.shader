#ifndef LBVH_SOLVER_SHADER
#define LBVH_SOLVER_SHADER

//#define USE_ALTERNATIVE_KERNEL_ARGS
//#define DEBUG_TREE_CREATION
//#define DEBUG_TRAVERSAL

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

#define LBVH_ROOT_NODE_MARKER       ((int)-1)

//The most significant bit(0x80000000) of a int32 is used to distinguish between leaf and internal nodes.
//If it is set, then the index is for an internal node; otherwise, it is a leaf node. 
//In both cases, the bit should be cleared to access the actual node index.
inline int isLeafNode(int index)
{
#ifdef DEBUG_TREE_CREATION
  return index != LBVH_ROOT_NODE_MARKER && (index - 1000000000) < 0;
#else
  return (index & 0x80000000) == 0;
#endif
}

inline int setInternalNodeMarker(int isLeaf, int index)
{
#ifdef DEBUG_TREE_CREATION
  return select(index + 1000000000, index, isLeaf);
#else
  return select((int)(index | 0x80000000), index, isLeaf);
#endif
}

inline int removeInternalNodeMarker(int index)
{
#ifdef DEBUG_TREE_CREATION
  return index - 1000000000;
#else
  return index & (~0x80000000);
#endif
}

inline int getCommonPrefixLength(const int2 left, const int2 right)
{
  int ret = clz(left.x ^ right.x);
  return select(ret, clz(left.y ^ right.y) + 32, ret == 32);
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
  const int                 nodeCount)
{
  const uint internalNodeCount = nodeCount - 1;
  const int internalNodeIndex = threadIndex();

  if (internalNodeIndex < internalNodeCount)
  {
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
      nodeParentNodeIndices[internalNodeIndex] = LBVH_ROOT_NODE_MARKER;
    }

    if (!leftIsLeaf)
    {
      nodeParentNodeIndices[splitPosition] = internalNodeIndex;
    }
    if (!rightIsLeaf)
    {
      nodeParentNodeIndices[splitPosition + 1] = internalNodeIndex;
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
#ifndef USE_ALTERNATIVE_KERNEL_ARGS
  Device XAB*               treeInternalNodeBoundingBoxes,
#else
  const Device XAB*         treeInternalNodeBoundingBoxesIn,
#endif
  Device uint*              visitedInternalNodes,
  const Device BVHNodeInfo* treeInternalNodes,
  const Device uint*        leafParentNodeIndices,
  const Device uint*        nodeParentNodeIndices,
  const Device XAB*         particleBoundingBoxes,
  const int                 nodeCount)
{
  int index = threadIndex();

#ifdef USE_ALTERNATIVE_KERNEL_ARGS
  Device XAB* treeInternalNodeBoundingBoxes = treeInternalNodeBoundingBoxesIn;
#endif

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
      currentNodeIndex = nodeParentNodeIndices[currentNodeIndex];

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

inline int intersectXAB(const XAB a, const XAB b)
{
  const int3 ret = (a.min < b.max) && (a.max > b.min);
  return (ret.x && ret.y && ret.z);
}

#define BVH_TRAVERSAL_FROM_PARENT   1
#define BVH_TRAVERSAL_FROM_CHILD    2
#define BVH_TRAVERSAL_FROM_SIBLING  3

#define INIT_POLL()     ushort poll_count = 0;
#define POLL_TIMEOUT()  (poll_count++ >= 20000)

inline float3 traverseBinaryTree(
  const ParticleStruct                currentParticle,
  const Device ParticleStruct*        particlesPredictedOld,
  const Device BVHNodeInfo*           treeInternalNodes,
  const Device uint*                  leafParentNodeIndices,
  const Device uint*                  nodeParentNodeIndices,
  const Device XAB*                   treeInternalNodeBoundingBoxes,
  const ParticleCollisionData         collisionData,
#ifdef MARK_COLLIDED_PARTICLES
  Device ParticleCollisionData*       particleCollisionData,
#else
  const Device ParticleCollisionData* particleCollisionData,
#endif
  const ParticleSharedData            sharedData,
  const int                           index)
{
#ifdef MARK_COLLIDED_PARTICLES
  bool collided = false;
#endif

  float3 output = constructFloat3(0.f);
  short collisionCount = 0;

  XAB particleBoundingBox;
  particleBoundingBox.min = currentParticle.position - constructFloat3(collisionData.radius);
  particleBoundingBox.max = currentParticle.position + constructFloat3(collisionData.radius);

  const float sdfMagnitude = length(collisionData.transformedSdfGradient);

  // mark index of the root node internal
  uint currentNodeIndex = setInternalNodeMarker(0, 0);
  uchar state = BVH_TRAVERSAL_FROM_PARENT;

  INIT_POLL();

  while (true && !POLL_TIMEOUT())
  {
    uint nextCurrentNodeIndex;

    // traverse while a leaf node is found
    while (!POLL_TIMEOUT())
    {
      bool switchBit;
      uint parentIndex;
      BVHNodeInfo parentNode;

      if (currentNodeIndex != setInternalNodeMarker(0, 0))
      {
        // fetch parent index
        parentIndex = select(nodeParentNodeIndices[removeInternalNodeMarker(currentNodeIndex)], leafParentNodeIndices[currentNodeIndex], isLeafNode(currentNodeIndex));

        // mark as internal node
        if (isLeafNode(parentIndex))
        {
          parentIndex = setInternalNodeMarker(0, parentIndex);
        }

        // parentNode is only needed if from parent or child
        if (state != BVH_TRAVERSAL_FROM_SIBLING)
        {
          parentNode = treeInternalNodes[removeInternalNodeMarker(parentIndex)];
        }
      }
      else
      {
        parentIndex = LBVH_ROOT_NODE_MARKER;
      }

#ifdef DEBUG_TRAVERSAL
      printf("Node: %d %d\n", currentNodeIndex, parentIndex);
#endif

      if (state == BVH_TRAVERSAL_FROM_CHILD)
      {
        if (currentNodeIndex == LBVH_ROOT_NODE_MARKER)
        {
          break;
        }

#ifdef DEBUG_TRAVERSAL
        if (currentNodeIndex == parentNode.child[0])
        {
          currentNodeIndex = parentNode.child[1];
          state = BVH_TRAVERSAL_FROM_SIBLING;
          printf("C->S: %d %d\n", currentNodeIndex, parentNode.child[1]);
        }
        else
        {
          printf("C->P: %d %d\n", currentNodeIndex, parentIndex);
          currentNodeIndex = parentIndex;
          state = BVH_TRAVERSAL_FROM_CHILD;
        }
#else
        switchBit = (currentNodeIndex == parentNode.child[0]);
        currentNodeIndex = select(parentIndex, parentNode.child[1], switchBit);
        state = select(BVH_TRAVERSAL_FROM_CHILD, BVH_TRAVERSAL_FROM_SIBLING, switchBit);
#endif
      }
      else // from parent or silbing
      {
        const XAB boundingBox = treeInternalNodeBoundingBoxes[select(removeInternalNodeMarker(currentNodeIndex), (int)currentNodeIndex, isLeafNode(currentNodeIndex))];

        // store the incoming state
        switchBit = (state == BVH_TRAVERSAL_FROM_SIBLING);

        // switch to next state
        state = select(BVH_TRAVERSAL_FROM_SIBLING, BVH_TRAVERSAL_FROM_CHILD, switchBit);

        if (intersectXAB(particleBoundingBox, boundingBox) == 0)
        {
#ifdef DEBUG_TRAVERSAL
          if (state == BVH_TRAVERSAL_FROM_SIBLING)
          {
            printf("S->C: %d %d\n", currentNodeIndex, parentIndex);
            currentNodeIndex = parentIndex;
            state = BVH_TRAVERSAL_FROM_CHILD;
          }
          else
          {
            printf("P->S: %d %d\n", currentNodeIndex, parentNode.child[1]);
            currentNodeIndex = parentNode.child[1];
            state = BVH_TRAVERSAL_FROM_SIBLING;
          }
#else
          currentNodeIndex = select(parentNode.child[1], parentIndex, switchBit);
#endif
        }
        else if (isLeafNode(currentNodeIndex))
        {
#ifdef DEBUG_TRAVERSAL
          if (state == BVH_TRAVERSAL_FROM_SIBLING)
          {
            nextCurrentNodeIndex = parentIndex;
            state = BVH_TRAVERSAL_FROM_CHILD;
            printf("S->C: %d %d\n", currentNodeIndex, nextCurrentNodeIndex);
          }
          else
          {
            nextCurrentNodeIndex = parentNode.child[1];
            state = BVH_TRAVERSAL_FROM_SIBLING;
            printf("P->S: %d %d\n", currentNodeIndex, nextCurrentNodeIndex);
          }
#else
          nextCurrentNodeIndex = select(parentNode.child[1], parentIndex, switchBit);
#endif
          break;
        }
        else
        {
          BVHNodeInfo node = treeInternalNodes[removeInternalNodeMarker(currentNodeIndex)];
#ifdef DEBUG_TRAVERSAL
          printf(" ->C: %d %d\n", currentNodeIndex, node.child[0]);
#endif
          currentNodeIndex = node.child[0];
          state = BVH_TRAVERSAL_FROM_PARENT;
        }
      }
    }

    if (currentNodeIndex == LBVH_ROOT_NODE_MARKER)
    {
      break;
    }

#ifdef DEBUG_TRAVERSAL
    printf("Test: %d %d\n", index, currentNodeIndex);
#endif
    // while ()
    // test colision if not an invalid node
    {
      output -= sharedData.collisionDamping * processParticleCollision(currentParticle, particlesPredictedOld[currentNodeIndex], collisionData, currentNodeIndex, sdfMagnitude,
#ifdef MARK_COLLIDED_PARTICLES
        particleCollisionData, &collided, &collisionCount);
#else
        particleCollisionData, &collisionCount);
#endif
    }

    if (nextCurrentNodeIndex == LBVH_ROOT_NODE_MARKER)
    {
      break;
    }

    currentNodeIndex = nextCurrentNodeIndex;
  }

#ifdef MARK_COLLIDED_PARTICLES
  particleCollisionData[index].radius = fabs(collisionData.radius) * (collided ? -1.f : 1.f);
#endif
//
//  if (collisionCount)
//  {
//    output /= collisionCount;
//  }

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
  Device ParticleStruct*              particles,
  Device ParticleStruct*              particles2,
  const Device ParticleStruct*        particlesOld,
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
  const Device ParticleAuxData*       particleAuxData,
  const Device PartitionInfo*         partitions,
  const Device EntityLocation*        entityLocation,
  Const PhySystemOffsets*             globalOffsets,
  const uint                          nodeCount)
{
#define batchMultiple 1

  volatile Shared int batchOffset[32];
  volatile Shared short batchCount[32];

  if (threadIndex() == 0)
  {
    atomicStore(batchCounter, 0);
  }

  localMemBarrier();

  const uchar subGroupLocalIndex = threadLocalIndex() & (COMPUTE_SUB_GROUP_SIZE - 1);
  const uchar subGroupIndex = threadLocalIndex() >> COMPUTE_SUB_GROUP_EXP;

  if (subGroupLocalIndex == 0)
  {
    batchOffset[subGroupIndex] = 0;
    batchCount[subGroupIndex] = 0;
  }

  // process until all batches are exhausted
  while (true)
  {
    if (subGroupLocalIndex == 0 && batchCount[subGroupIndex] == 0)
    {
      batchOffset[subGroupIndex] = atomicAdd(batchCounter, COMPUTE_SUB_GROUP_SIZE * batchMultiple);
      batchCount[subGroupIndex] = COMPUTE_SUB_GROUP_SIZE * batchMultiple;
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
      ParticleStruct currentParticle = particlesOld[index];
      const IdentityInfo identity = currentParticle.identity;
      ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(identity);
      const PhySystemOffsets phySystemOffsets = globalOffsets[nodeIdentity.solverType];

      nodeIdentity.entityId += phySystemOffsets.globalSolverOffset;
      nodeIdentity.instanceId += phySystemOffsets.globalInstanceOffset;

      const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];
      const ParticleNodeLocator nodeLocator = getNodeLocator(index, phySystemOffsets.globalNodeOffset + partitions[nodeIdentity.instanceId].offset, entityLocation[nodeIdentity.entityId].node);

      const ParticleCollisionData collisionData = particleCollisionData[index];

      // find position change due to collision
      float3 delta = traverseBinaryTree(
        currentParticle,
        particlesOld,
        treeInternalNodes,
        leafParentNodeIndices,
        nodeParentNodeIndices,
        treeInternalNodeBoundingBoxes,
        collisionData,
        particleCollisionData,
        sharedData,
        index);

      // update position
      currentParticle.position += delta;

      // apply boundary
      boundaryCollision(&currentParticle, particles2, nodeLocator, collisionData);

      // save updated position
      currentParticle.identity = identity;
      particles[index] = currentParticle;

      if (particles2)
      {
        particles2[index].position += delta;
        particles2[index].identity = identity;
      }
    }

    if (subGroupLocalIndex == 0)
    {
      batchOffset[subGroupIndex] += COMPUTE_SUB_GROUP_SIZE;
      batchCount[subGroupIndex] -= COMPUTE_SUB_GROUP_SIZE;
    }
  }
}

#endif
