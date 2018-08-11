#ifndef LBVH_SOLVER_SHADER
#define LBVH_SOLVER_SHADER

uint commonPrefixLength(uint i[2], uint j[2])
{
  uint ret = clz(i[0] ^ j[0]);
  return ret ? ret : clz(i[1] ^ j[1]);
}

void commonPrefix(Shared uint prefix[2], const uint i[2], const uint j[2])
{
  // This function only needs to return (i & j) in order for the algorithm to work,
  // but it may help with debugging to mask out the lower bits.
  uint length = commonPrefixLength(i, j);
  uint sharedBits[2] = { i[0] & j[0], i[1] & j[1] };
  uint bitmask[2] = { (~0) << (64 - (length - 32)), (~0) << (64 - length) }; //Set all bits after the common prefix to 0

  prefix[0] = sharedBits[0] & bitmask[0];
  prefix[1] = sharedBits[1] & bitmask[1];
}

void getCommonPrefixAtIndex(Shared uint commonPrefixInfo[3], const Shared uint *localBvhLeafs, const uint index, const uint localIndexFrom1)
{
  //Here, (internalNodeIndex + 1) is never out of bounds since it is a leaf node index,
  //and the number of internal nodes is always numLeafNodes - 1

  //Binary radix tree construction algorithm does not work if there are duplicate morton codes.
  //Append the index of each leaf node to each morton code so that there are no duplicates.
  //The algorithm also requires that the morton codes are sorted in ascending order; this requirement
  //is also satisfied with this method, as (leftLeafIndex < rightLeafIndex) is always true.
  //
  //upsample(a, b) == ( ((b3Int64)a) << 32) | b
  uint nonduplicateLeftMortonCode[2] = { localBvhLeafs[localIndexFrom1], index };
  uint nonduplicateRightMortonCode[2] = { localBvhLeafs[localIndexFrom1 + 1], index + 1 };

  //out_commonPrefixes[internalNodeIndex] = computeCommonPrefix(nonduplicateLeftMortonCode, nonduplicateRightMortonCode);
  //out_commonPrefixLengths[internalNodeIndex] = computeCommonPrefixLength(nonduplicateLeftMortonCode, nonduplicateRightMortonCode);

  commonPrefix(commonPrefixInfo, nonduplicateLeftMortonCode, nonduplicateRightMortonCode);
  commonPrefixInfo[2] = commonPrefixLength(nonduplicateLeftMortonCode, nonduplicateRightMortonCode);
}

uint getIndexWithInternalNodeMarkerSet(uint isLeaf, uint index)
{
  return index | (isLeaf << 31);
}

Kernel void findSplitKernel(
  Device BVHLeafInfo* bvhLeafs,
  Device BVHNodeInfo* bvhNodes,
  Device uint*        planeIndex,
  const uint          length)
{
  const uint index = threadIndex();
  const uint localIndexFrom1 = threadLocalIndex() + 1;
  const uint groupIndex = threadGroupIndex();

  Shared uint localBvhLeafs[COMPUTE_MAX_THREADS + 2];
  Shared uint commonPrefixInfo[COMPUTE_MAX_THREADS + 1][3];

  if (index < length)
  {
    uint localCommonPrefix[2];
    uint localCommonPrefixLength;

    localBvhLeafs[localIndexFrom1] = bvhLeafs[index].mortonCode;
    if (index == (length - 1) || localIndexFrom1 == COMPUTE_MAX_THREADS)
    {
      localBvhLeafs[localIndexFrom1 + 1] = bvhLeafs[index + 1].mortonCode;
    }

    localMemBarrier();

    if (index < (length - 1))
    {
      if (groupIndex > 0 && localIndexFrom1 == 1)
      {
        getCommonPrefixAtIndex(commonPrefixInfo, localBvhLeafs, index - 1, localIndexFrom1 - 1);
      }
      getCommonPrefixAtIndex(commonPrefixInfo, localBvhLeafs, index, localIndexFrom1);

      /*//Here, (internalNodeIndex + 1) is never out of bounds since it is a leaf node index,
      //and the number of internal nodes is always numLeafNodes - 1
      uint leftLeafIndex = index;
      uint rightLeafIndex = index + 1;

      uint leftLeafMortonCode = bvhLeafs[localIndex].mortonCode;
      uint rightLeafMortonCode = bvhLeafs[localIndex + 1].mortonCodes;

      //Binary radix tree construction algorithm does not work if there are duplicate morton codes.
      //Append the index of each leaf node to each morton code so that there are no duplicates.
      //The algorithm also requires that the morton codes are sorted in ascending order; this requirement
      //is also satisfied with this method, as (leftLeafIndex < rightLeafIndex) is always true.
      //
      //upsample(a, b) == ( ((b3Int64)a) << 32) | b
      uint nonduplicateLeftMortonCode[2] = { leftLeafMortonCode, leftLeafIndex };
      uint nonduplicateRightMortonCode[2] = { rightLeafMortonCode, rightLeafIndex };

      //out_commonPrefixes[internalNodeIndex] = computeCommonPrefix(nonduplicateLeftMortonCode, nonduplicateRightMortonCode);
      //out_commonPrefixLengths[internalNodeIndex] = computeCommonPrefixLength(nonduplicateLeftMortonCode, nonduplicateRightMortonCode);

      commonPrefix(localCommonPrefix, nonduplicateLeftMortonCode, nonduplicateRightMortonCode);
      localCommonPrefixLength = commonPrefix(nonduplicateLeftMortonCode, nonduplicateRightMortonCode);*/
    }

    uint numInternalNodes = length - 1;

    uint leftSplitIndex = index - 1;
    uint rightSplitIndex = index;

    uint leftCommonPrefix = (leftSplitIndex >= 0) ? commonPrefixInfo[localIndexFrom1][2] : -1;
    uint rightCommonPrefix = (rightSplitIndex < (length - 1)) ? commonPrefixInfo[localIndexFrom1 + 1][2] : -1;

    // Parent node is the highest adjacent common prefix that is lower than the node's common prefix
    // Leaf nodes are considered as having the highest common prefix
    uint isLeftHigherCommonPrefix = (leftCommonPrefix > rightCommonPrefix);

    // Handle cases for the edge nodes; the first and last node
    // For leaf nodes, leftCommonPrefix and rightCommonPrefix should never both be B3_PLBVH_INVALID_COMMON_PREFIX
    if (leftCommonPrefix == -1)
    {
      isLeftHigherCommonPrefix = false;
    }

    if (rightCommonPrefix == -1)
    {
      isLeftHigherCommonPrefix = true;
    }

    uint parentNodeIndex = (isLeftHigherCommonPrefix) ? leftSplitIndex : rightSplitIndex;
    uint isRightChild = isLeftHigherCommonPrefix; // If the left node is the parent, then this node is its right child and vice versa
    uint isLeaf = 1;

    bvhNodes[parentNodeIndex].node[isRightChild] = getIndexWithInternalNodeMarkerSet(isLeaf, index);
  }
}

/*
@kernel Apply boundary constrain.
@param particles Initial particle position.
@param particlesPredicted Integrated particle position.
@param particleIdentities Particle identifiers.
@param particleSharedData Particle entity shared data.
@param particleAuxData Additional particle data.
@param partitions Instance partition data.
@param entityLocation Buffer containing entity boundary info.
@param nodeCount Total nodes in the solver.
*/
Kernel void boundaryCollisionKernel(
  Device ParticleStruct*            particles,
  Device ParticleStruct*            particlesPredicted,
  const Device IdentityInfo*        particleIdentities,
  const Device ParticleSharedData*  particleSharedData,
  const Device ParticleAuxData*     particleAuxData,
  const Device PartitionInfo*       partitions,
  const Device EntityLocation*      entityLocation,
  Const uint4*                      globalOffsets,
  const uint                        nodeCount)
{
  const uint index = threadIndex();
  const uint localIndex = threadLocalIndex();

  if (index < nodeCount)
  {
    const IdentityInfo identity = particleIdentities[index];

    const uint solverType = getSolverType(identity);
    const uint globalSolverOffset = globalOffsets[solverType].z;
    const uint globalNodeOffset = globalOffsets[solverType].x;
    const uint globalInstanceOffset = globalOffsets[solverType].y;

    const uint entityId = globalSolverOffset + getEntityId(identity);
    const uint instanceId = globalInstanceOffset + getInstanceId(identity);

    const ParticleSharedData sharedData = particleSharedData[entityId];

    const uint absoluteNodeOffset = globalNodeOffset + partitions[instanceId].offset;
    const uint relativeNodeIndex = index % entityLocation[entityId].node.count;
    const uint absoluteNodeIndex = absoluteNodeOffset + relativeNodeIndex;

    const float invMass = getInvMass(&sharedData, particleAuxData, entityLocation[entityId].node.offset + relativeNodeIndex);

    if (invMass) // only if movable
    {
      float dely = 0.f;

      if (particlesPredicted[absoluteNodeIndex].position.y <= -2.f)
      {
        dely = -2.f - particlesPredicted[absoluteNodeIndex].position.y;
        particles[absoluteNodeIndex].position.y += dely;
        particlesPredicted[absoluteNodeIndex].position.y += dely;
      }
    }
  }
}

#endif