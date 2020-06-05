#ifndef LBVH_SOLVER_SHADER
#define LBVH_SOLVER_SHADER

//#define DEBUG_TREE_CREATION
//#define DEBUG_TRAVERSAL

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
  constantKernelInput(uint,     nodeCount)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint index = threadIndex();

  if (index < nodeCount)
  {
    const XAB mergedBox = mergedParticleBoundingBox[0];
    const float3 inverseMergedBoxSize = 1024.f / (mergedBox.max - mergedBox.min);
    const float3 mergedBoxCenter = (mergedBox.min + mergedBox.max) * 0.5f;

    const ParticleStruct particle = particles[index];

    // Quantize into integer coordinates
    // floor() is needed to prevent the center cell, at (0,0,0) from being twice the size
    float3 positionRelativeToCenter = (particle.position - mergedBoxCenter) * inverseMergedBoxSize;

    int3 quantizedPosition = convertInt3(select(floor(positionRelativeToCenter), positionRelativeToCenter, positionRelativeToCenter >= 0.0f));

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
  constantKernelInput(int,  nodeCount)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const int internalNodeCount = nodeCount - 1;
  const int internalNodeIndex = threadIndex();

  if (internalNodeIndex < internalNodeCount)
  {
    // code of leaf before this
    int2 prevNodePrefix = select(constructInt2(0xFFFFFFFF), constructInt2(bvhLeafs[internalNodeIndex - 1].mortonCode, internalNodeIndex - 1), selectInput2(internalNodeIndex > 0));
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
  Device XAB*               treeInternalNodeBoundingBoxes,
  atomicKernelInput(uint,   visitedInternalNodes),
  const Device BVHNodeInfo* treeInternalNodes,
  const Device uint*        leafParentNodeIndices,
  const Device uint*        nodeParentNodeIndices,
  const Device XAB*         particleBoundingBoxes,
  constantKernelInput(int,  nodeCount)
  KERNEL_GLOBAL_ARGUMENTS)
{
  int index = threadIndex();

  if (index < nodeCount)
  {
    // process leaf node bounding boxes first
    uint currentNodeIndex = leafParentNodeIndices[index];
    // get the processing order
    uint visited = atomicAdd(&visitedInternalNodes[currentNodeIndex], 1);
    // get internal node for the leaf
    BVHNodeInfo internalNode = treeInternalNodes[currentNodeIndex];

    // only process if leaf
    while (visited)
    {
      // bounding box accumulated by the thread
      XAB mergedBoundingBox;
      clearXAB(&mergedBoundingBox, INFINITY);

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
      visited = atomicAdd(&visitedInternalNodes[currentNodeIndex], 1);
    }
  }
}

inline int intersectXAB(const Thread XAB* a, const Thread XAB* b)
{
  const int3 ret = constructInt3(a->min < b->max) & constructInt3(a->max > b->min);
  return (ret.x && ret.y && ret.z);
}

#define BVH_TRAVERSAL_FROM_PARENT   1
#define BVH_TRAVERSAL_FROM_CHILD    2
#define BVH_TRAVERSAL_FROM_SIBLING  3

inline float3 stacklessTraverseBinaryTree(
  const Thread ParticleStruct*        currentParticle,
  const Device ParticleStruct*        particlesPredictedOld,
  const ParticleDifferential          selfParticleDiff,
  const Device ParticleDifferential*  particlesDiff,
  const Device BVHNodeInfo*           treeInternalNodes,
  const Device uint*                  leafParentNodeIndices,
  const Device uint*                  nodeParentNodeIndices,
  const Device XAB*                   treeInternalNodeBoundingBoxes,
  const Thread ParticleCollisionData* collisionData,
  const uint                          stablizationPass,
  const short                         solverType,
  Device ParticleStruct*              particlesDelta,
#ifdef MARK_COLLIDED_PARTICLES
  Device ParticleCollisionData*       particleCollisionData,
#else
  const Device ParticleCollisionData* particleCollisionData,
#endif
  const Thread CollisionSolverData*   collisionSolverData,
  const int                           index)
{
#ifdef MARK_COLLIDED_PARTICLES
  bool collided = false;
#endif

  float3 output = constructFloat3(0.f);
  uint collisionCount = 0;

  XAB particleBoundingBox;
  particleBoundingBox.min = currentParticle->position - constructFloat3(collisionData->radius);
  particleBoundingBox.max = currentParticle->position + constructFloat3(collisionData->radius);

  const float sdfMagnitude = collisionData->gradientMagnitude;

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
      printf("Node: %d %d %d\n", index, currentNodeIndex, parentIndex);
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
          printf("C->S: %d %d %d\n", index, currentNodeIndex, parentNode.child[1]);
          currentNodeIndex = parentNode.child[1];
          state = BVH_TRAVERSAL_FROM_SIBLING;
        }
        else
        {
          printf("C->P: %d %d %d\n", index, currentNodeIndex, parentIndex);
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

#ifndef DEBUG_TRAVERSAL
        // store the incoming state
        switchBit = (state == BVH_TRAVERSAL_FROM_SIBLING);

        // switch to next state
        state = select(BVH_TRAVERSAL_FROM_SIBLING, BVH_TRAVERSAL_FROM_CHILD, switchBit);
#endif

        // leaf test has to be done before intersect XAB so that all leaf siblings are processed else it may get skipped
        if (isLeafNode(currentNodeIndex))
        {
#ifdef DEBUG_TRAVERSAL
          if (state == BVH_TRAVERSAL_FROM_SIBLING)
          {
            printf("S->C: %d %d %d\n", index, currentNodeIndex, parentIndex);
            nextCurrentNodeIndex = parentIndex;
            state = BVH_TRAVERSAL_FROM_CHILD;
          }
          else
          {
            printf("P->S: %d %d %d\n", index, currentNodeIndex, parentNode.child[1]);
            nextCurrentNodeIndex = parentNode.child[1];
            state = BVH_TRAVERSAL_FROM_SIBLING;
          }
#else
          nextCurrentNodeIndex = select(parentNode.child[1], parentIndex, switchBit);
#endif
          break;
        }
        else if (intersectXAB(&particleBoundingBox, &boundingBox) == 0)
        {
#ifdef DEBUG_TRAVERSAL
          if (state == BVH_TRAVERSAL_FROM_SIBLING)
          {
            printf("S->C: %d %d %d\n", index, currentNodeIndex, parentIndex);
            currentNodeIndex = parentIndex;
            state = BVH_TRAVERSAL_FROM_CHILD;
          }
          else
          {
            printf("P->S: %d %d %d\n", index, currentNodeIndex, parentNode.child[1]);
            currentNodeIndex = parentNode.child[1];
            state = BVH_TRAVERSAL_FROM_SIBLING;
          }
#else
          currentNodeIndex = select(parentNode.child[1], parentIndex, switchBit);
#endif
        }
        else
        {
          BVHNodeInfo node = treeInternalNodes[removeInternalNodeMarker(currentNodeIndex)];
#ifdef DEBUG_TRAVERSAL
          printf(" ->C: %d %d %d\n", index, currentNodeIndex, node.child[0]);
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
    if (index != currentNodeIndex)
    {
      const ParticleStruct otherParticle = particlesPredictedOld[currentNodeIndex];
      if (shouldCheckForCollision(solverType, index, currentNodeIndex, currentParticle, &otherParticle))
      {
        const ParticleDifferential otherParticleDiff = particlesDiff[currentNodeIndex];
        output += processParticleCollision(currentParticle, &selfParticleDiff, &otherParticle, &otherParticleDiff, true,
          collisionData, collisionSolverData, currentNodeIndex, index, sdfMagnitude, &collisionCount, stablizationPass, solverType, particlesDelta,
#ifdef MARK_COLLIDED_PARTICLES
          particleCollisionData, &collided);
#else
          particleCollisionData);
#endif
      }
    }

    if (nextCurrentNodeIndex == LBVH_ROOT_NODE_MARKER)
    {
      break;
    }

    currentNodeIndex = nextCurrentNodeIndex;
  }

#ifdef MARK_COLLIDED_PARTICLES
  particleCollisionData[index].radius = fabs(collisionData->radius) * (collided ? -1.f : 1.f);
#endif

  if (collisionCount)
  {
    output /= collisionCount;
  }

  return output;
}

inline float3 stackTraverseBinaryTree(
  const Thread ParticleStruct*        currentParticle,
  const Device ParticleStruct*        particlesPredictedOld,
  const ParticleDifferential          selfParticleDiff,
  const Device ParticleDifferential*  particlesDiff,
  const Device BVHNodeInfo*           treeInternalNodes,
  const Device uint*                  leafParentNodeIndices,
  const Device uint*                  nodeParentNodeIndices,
  const Device XAB*                   treeInternalNodeBoundingBoxes,
  const Thread ParticleCollisionData* collisionData,
  const uint                          stablizationPass,
  const short                         solverType,
  Device ParticleStruct*              particlesDelta,
#ifdef MARK_COLLIDED_PARTICLES
  Device ParticleCollisionData*       particleCollisionData,
#else
  const Device ParticleCollisionData* particleCollisionData,
#endif
  const Thread CollisionSolverData*   collisionSolverData,
  const int                           index)
{
  uchar stackTop = 0;
  uint traversalStack[64];

#ifdef MARK_COLLIDED_PARTICLES
  bool collided = false;
#endif

  float3 output = constructFloat3(0.f);
  uint collisionCount = 0;

  XAB particleBoundingBox;
  particleBoundingBox.min = currentParticle->position - constructFloat3(collisionData->radius);
  particleBoundingBox.max = currentParticle->position + constructFloat3(collisionData->radius);

  const float sdfMagnitude = collisionData->gradientMagnitude;

  // mark index of the root node internal
  uint currentNodeIndex = setInternalNodeMarker(0, 0);

  INIT_POLL();

  while (!POLL_TIMEOUT())
  {
#ifdef DEBUG_TRAVERSAL
    printf("Node: %d %d %d\n", index, currentNodeIndex, isLeafNode(currentNodeIndex));
#endif

    // traverse while a leaf node is found
    while (!isLeafNode(currentNodeIndex) && !POLL_TIMEOUT())
    {
      const BVHNodeInfo node = treeInternalNodes[removeInternalNodeMarker(currentNodeIndex)];

      const XAB leftBoundingBox = treeInternalNodeBoundingBoxes[select(removeInternalNodeMarker(node.child[0]), (int)node.child[0], isLeafNode(node.child[0]))];
      const XAB rightBoundingBox = treeInternalNodeBoundingBoxes[select(removeInternalNodeMarker(node.child[1]), (int)node.child[1], isLeafNode(node.child[1]))];

      if (intersectXAB(&particleBoundingBox, &leftBoundingBox))
      {
        traversalStack[stackTop++] = node.child[0];
#ifdef DEBUG_TRAVERSAL
        printf("Stack Push: %d %d\n", index, node.child[0]);
#endif
      }
      if (intersectXAB(&particleBoundingBox, &rightBoundingBox))
      {
        traversalStack[stackTop++] = node.child[1];
#ifdef DEBUG_TRAVERSAL
        printf("Stack Push: %d %d\n", index, node.child[1]);
#endif
      }

      // break if the stack is empty
      if (stackTop == 0)
      {
        // mark node invalid
        currentNodeIndex = LBVH_ROOT_NODE_MARKER;
        break;
      }

      currentNodeIndex = traversalStack[--stackTop];
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
      const ParticleStruct otherParticle = particlesPredictedOld[currentNodeIndex];
      if (shouldCheckForCollision(solverType, index, currentNodeIndex, currentParticle, &otherParticle))
      {
        const ParticleDifferential otherParticleDiff = particlesDiff[currentNodeIndex];
        output += processParticleCollision(currentParticle, &selfParticleDiff, &otherParticle, &otherParticleDiff, true,
          collisionData, collisionSolverData, currentNodeIndex, index, sdfMagnitude, &collisionCount, stablizationPass, solverType, particlesDelta,
#ifdef MARK_COLLIDED_PARTICLES
          particleCollisionData, &collided);
#else
          particleCollisionData);
#endif
      }
    }

    // exit if nothing to fetch
    if (stackTop == 0)
    {
      break;
    }

    // pop from the stack
    currentNodeIndex = traversalStack[--stackTop];
  }

#ifdef MARK_COLLIDED_PARTICLES
  particleCollisionData[index].radius = fabs(collisionData->radius) * (collided ? -1.f : 1.f);
#endif

  if (collisionCount)
  {
    output /= collisionCount;
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
@param systemSettings Settings for the physics system.
@param gridParticleCellIndex Computed cell index for each particle.
@param nodeCount Total nodes in the solver.
@param occupiedCellCount Total active grid cells.
*/
Kernel void applyCollisions(
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
  constantKernelInput(uint,           stablizationPass)
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
}

#endif
