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
  const Device ParticleStruct*      particlesPredicted,
  const Device ParticleSharedData*  particleSharedData,
  const Device ParticleAuxData*     particleAuxData,
  const Device PartitionInfo*       partitions,
  const Device EntityLocation*      entityLocation,
  Const PhySystemOffsets*           globalOffsets,
  const uint                        nodeCount)
{
  const uint index = threadIndex();
  if (index < nodeCount)
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

    int3 quantizedPosition;
    quantizedPosition.x = (int)((positionRelativeToCenter.x >= 0.0f) ? positionRelativeToCenter.x : floor(positionRelativeToCenter.x));
    quantizedPosition.y = (int)((positionRelativeToCenter.y >= 0.0f) ? positionRelativeToCenter.y : floor(positionRelativeToCenter.y));
    quantizedPosition.z = (int)((positionRelativeToCenter.z >= 0.0f) ? positionRelativeToCenter.z : floor(positionRelativeToCenter.z));

    // Clamp coordinates into [-512, 511], then convert range from [-512, 511] to [0, 1023]
    quantizedPosition = max(-512, min(quantizedPosition, 511)) + 512;

    //Interleave bits(assign a morton code, also known as a z-curve)
    BVHLeafInfo bvhLeaf;
    bvhLeaf.mortonCode = get32BitMortonCode(quantizedPosition);
    bvhLeaf.index = index;

    bvhLeafs[index] = bvhLeaf;
  }
}

//The most significant bit(0x80000000) of a int32 is used to distinguish between leaf and internal nodes.
//If it is set, then the index is for an internal node; otherwise, it is a leaf node. 
//In both cases, the bit should be cleared to access the actual node index.
int isLeafNode(int index)
{
  return (index - 1000000000) < 0;
  //return (index >> 31) == 0;
}

int setInternalNodeMarker(int isLeaf, int index)
{
  return isLeaf ? index : (index + 1000000000);
  //return isLeaf ? index : (index | 0x80000000);
}

int removeInternalNodeMarker(int index)
{
  return index - 1000000000;
  //return index & (~0x80000000);
}

int getCommonPrefixLength(const uint2 left, const uint2 right)
{
  uint ret = clz(left.x ^ right.x);
  return (ret == 32) ? (clz(left.y ^ right.y) + 32) : ret;
}

#define LBVH_INVALID_COMMON_PREFIX  ((int)128)
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

    const uint2 nodePrefix = leafNodeCommonPrefixes[internalNodeIndex];
    const int nodePrefixLength = leafNodeCommonPrefixLengths[internalNodeIndex];

    int leftIndex = -1;
    int rightIndex = -1;

    // Find nearest element to left with a lower common prefix
    for (int i = internalNodeIndex - 1; i >= 0; i--)
    {
      int nodeLeftSharedPrefixLength = getSharedPrefixLength(nodePrefix, leafNodeCommonPrefixes[i], nodePrefixLength, leafNodeCommonPrefixLengths[i]);
      if (nodeLeftSharedPrefixLength < nodePrefixLength)
      {
        leftIndex = i;
        break;
      }
    }

    // Find nearest element to right with a lower common prefix
    for (int i = internalNodeIndex + 1; i < internalNodeCount; i++)
    {
      int nodeRightSharedPrefixLength = getSharedPrefixLength(nodePrefix, leafNodeCommonPrefixes[i], nodePrefixLength, leafNodeCommonPrefixLengths[i]);
      if (nodeRightSharedPrefixLength < nodePrefixLength)
      {
        rightIndex = i;
        break;
      }
    }

    // Select parent
    {
      const int leftPrefixLength = (leftIndex != -1) ? leafNodeCommonPrefixLengths[leftIndex] : LBVH_INVALID_COMMON_PREFIX;
      const int rightPrefixLength = (rightIndex != -1) ? leafNodeCommonPrefixLengths[rightIndex] : LBVH_INVALID_COMMON_PREFIX;

      uint isLeftHigherPrefixLength = (leftPrefixLength > rightPrefixLength);

      if (leftPrefixLength == LBVH_INVALID_COMMON_PREFIX)       isLeftHigherPrefixLength = 0;
      else if (rightPrefixLength == LBVH_INVALID_COMMON_PREFIX) isLeftHigherPrefixLength = 1;

      const int parentNodeIndex = (isLeftHigherPrefixLength) ? leftIndex : rightIndex;

      const uint isRootNode = (leftIndex == -1 && rightIndex == -1);
      internalNodeParentIndices[internalNodeIndex] = (!isRootNode) ? parentNodeIndex : LBVH_ROOT_NODE_MARKER;

      if (!isRootNode)
      {
        int isRightChild = (isLeftHigherPrefixLength);  //If the left node is the parent, then this node is its right child and vice versa

        Device int* childNodesAsInt = (Device int*)&internalNodeChildIndices[parentNodeIndex];
        childNodesAsInt[isRightChild] = setInternalNodeMarker(0, internalNodeIndex);
      }
      else
      {
        out_rootNodeIndex[0] = setInternalNodeMarker(0, internalNodeIndex);
      }
    }

#else

    // code of leaf before this
    uint2 prevNodePrefix;
    if (internalNodeIndex > 0)
    {
      prevNodePrefix = constructUint2(bvhLeafs[internalNodeIndex - 1].mortonCode, internalNodeIndex - 1);
    }
    // code of this leaf
    const uint2 currNodePrefix = constructUint2(bvhLeafs[internalNodeIndex].mortonCode, internalNodeIndex);
    // code of leaf after this
    const uint2 nextNodePrefix = constructUint2(bvhLeafs[internalNodeIndex + 1].mortonCode, internalNodeIndex + 1);

    const int nodeLeftSharedPrefixLength = (internalNodeIndex > 0) ? getCommonPrefixLength(currNodePrefix, prevNodePrefix) : -1;
    const int nodeRightSharedPrefixLength = getCommonPrefixLength(currNodePrefix, nextNodePrefix);

    const int direction = (nodeRightSharedPrefixLength > nodeLeftSharedPrefixLength) ? 1 : -1;
    const int minLength = (nodeRightSharedPrefixLength > nodeLeftSharedPrefixLength) ? nodeLeftSharedPrefixLength : nodeRightSharedPrefixLength;

    int maxStep = 128;
    int nextInternalNodeIndex = internalNodeIndex + maxStep * direction;

    // Compute upper bound for the length of the range
    while (nextInternalNodeIndex >= 0 && nextInternalNodeIndex < nodeCount &&
      getCommonPrefixLength(currNodePrefix, constructUint2(bvhLeafs[nextInternalNodeIndex].mortonCode, nextInternalNodeIndex)) > minLength)
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
        getCommonPrefixLength(currNodePrefix, constructUint2(bvhLeafs[nextInternalNodeIndex].mortonCode, nextInternalNodeIndex)) > minLength)
      {
        endPosition += currentStep;
      }
      currentStep >>= 1;
    }

    int otherInternalNodeIndex = internalNodeIndex + endPosition * direction;
    const int splitLength = getCommonPrefixLength(currNodePrefix, constructUint2(bvhLeafs[otherInternalNodeIndex].mortonCode, otherInternalNodeIndex));

    int splitPosition = 0;

    // Find the split position using binary search
    do
    {
      endPosition = (endPosition >> 1) + (endPosition & 1);
      nextInternalNodeIndex = internalNodeIndex + (endPosition + splitPosition) * direction;
      if (nextInternalNodeIndex >= 0 && nextInternalNodeIndex < nodeCount &&
        getCommonPrefixLength(currNodePrefix, constructUint2(bvhLeafs[nextInternalNodeIndex].mortonCode, nextInternalNodeIndex)) > splitLength)
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
      mergedBoundingBox.min = INFINITY;
      mergedBoundingBox.max = -INFINITY;

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

#endif