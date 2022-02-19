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
  return index < 1000000000;
#else
  return asUshort2(index).y < 0x7FFF;
  //return (index & 0x80000000) == 0;
  //return index < 0x7FFFFFFF;
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
  return index & 0x7FFFFFFF;
#endif
}

inline short getBVHCommonPrefixLength(const uint2 left, const uint2 right)
{
  short ret = clz(left.x ^ right.x);
  return select(ret, (short)(clz(left.y ^ right.y) + 32), (short)(ret == 32));
}

inline short getNodesCommonPrefixLength(const Device BVHLeafInfo* bvhLeafs, const int currNodeIndex, const int nodeCount, const uint2 baseNode)
{
  if (currNodeIndex < 0 || currNodeIndex >= nodeCount)
  {
    return -1;
  }

  return getBVHCommonPrefixLength(baseNode, constructUint2(bvhLeafs[currNodeIndex].mortonCode, currNodeIndex));
}


/*
@kernel Create single array composed of all the primitives.
@param finalVertexArray Buffer containing all positions.
@param finalAttributeArray Buffer containing primitive attribute data.
@param finalVertexAttributeArray Buffer containing vertex attribute data.
@param primitiveBuffer Buffer containing primitive positions.
@param primitivePackingInfo Packing information for primitive structure.
@param attributeBuffer Buffer containing attribute inside a structure.
@param attributePackingInfo Packing information for attribute in primitive structure.
@param vertexAttribBuffer Buffer containing vertex attribute inside a structure.
@param vertexAttribPackingInfo Packing information for vertex structure.
@param primitiveBatchSize Primitives processed per thread.
@param primitiveCount Total primitives in the buffer.
@param primitiveType Primitive type for the dispatch.
@param vertexOffset Starting offset for storing vertex data.
*/
Kernel void collectPrimitives(
  Device PrimitiveStruct*           finalVertexArray,
  Device PrimitiveAttrib*           finalAttributeArray,
  Device VertexAttrib*              finalVertexAttributeArray,
  const Device PrimitiveStruct*     primitiveBuffer,
  constantKernelInput(PackingInfo,  primitivePackingInfo),
  const Device float*               attributeBuffer,
  constantKernelInput(PackingInfo,  attributePackingInfo),
  const Device float*               vertexAttribBuffer,
  constantKernelInput(PackingInfo,  vertexAttribPackingInfo),
  constantKernelInput(IdentityInfo, primitiveIdentity),
  constantKernelInput(uint,         primitiveBatchSize),
  constantKernelInput(uint,         primitiveCount),
  constantKernelInput(uint,         primitiveType),
  constantKernelInput(uint,         primitiveOffset),
  constantKernelInput(uint,         vertexOffset),
  constantKernelInput(float4x4,     matrix)
  KERNEL_THREAD_ARGUMENTS
  KERNEL_THREADGROUP_ARGUMENTS)
{
  const Device PrimitiveAttrib* primitiveAttribPtr  = (const Device PrimitiveAttrib*)extractPackedPointer(attributeBuffer, attributePackingInfo);
  const Device VertexAttrib* vertexAttribPtr = (const Device VertexAttrib*)extractPackedPointer(vertexAttribBuffer, vertexAttribPackingInfo);

  uint index = threadLocalIndex() + primitiveBatchSize * threadGroupIndex() * threadGroupSize();
  for (short b = 0; index < primitiveCount && b < primitiveBatchSize; index += threadGroupSize(), b++)
  {
    if (primitiveType == PrimitiveSphere)
    {
      PrimitiveStruct outPrim  = primitiveBuffer[index + primitivePackingInfo.elementOffset];

      outPrim.position = mulMatrixVec(matrix, constructFloat4(outPrim.position, 1.f)).xyz;
      outPrim.identity = primitiveIdentity;
      finalVertexArray[index + vertexOffset] = outPrim;
      finalAttributeArray[index + vertexOffset].radius = primitiveAttribPtr[index].radius;
    }
    else if (primitiveType == PrimitiveIndexedTriangle || primitiveType == PrimitiveTriangle)
    {
      uint3 vertIndices;
      if (primitiveType == PrimitiveIndexedTriangle)
      {
        vertIndices = primitiveAttribPtr[index].triangleIndex;
      }
      else
      {
        vertIndices = constructUint3(0, 1, 2) + index * 3;
        // for indexed array a non zero stride is assumed
        if (attributePackingInfo.strideIn4Bytes > 0)
        {
          vertIndices = primitiveAttribPtr[index].triangleIndex;
        }
      }

      // get vertex zero and vertex position
      PrimitiveStruct vert0 = primitiveBuffer[vertIndices.x];
      PrimitiveStruct vert1 = primitiveBuffer[vertIndices.y];
      PrimitiveStruct vert2 = primitiveBuffer[vertIndices.z];

      vert0.position = mulMatrixVec(matrix, constructFloat4(vert0.position, 1.f)).xyz;
      vert1.position = mulMatrixVec(matrix, constructFloat4(vert1.position, 1.f)).xyz;
      vert2.position = mulMatrixVec(matrix, constructFloat4(vert2.position, 1.f)).xyz;

      const VertexAttrib vertAttrib0 = vertexAttribPtr[vertIndices.x];
      const VertexAttrib vertAttrib1 = vertexAttribPtr[vertIndices.y];
      const VertexAttrib vertAttrib2 = vertexAttribPtr[vertIndices.z];

      if (primitiveType == PrimitiveIndexedTriangle)
      {
        vertIndices += vertexOffset;
      }
      else
      {
        vert1.position = vert1.position - vert0.position;
        vert2.position = vert2.position - vert0.position;
        vertIndices = vertexOffset + constructUint3(0, 1, 2) + index * 3;
      }

      vert0.identity = primitiveIdentity;
      vert1.identity = primitiveIdentity;
      vert2.identity = primitiveIdentity;

      finalVertexArray[vertIndices.x] = vert0;
      finalVertexArray[vertIndices.y] = vert1;
      finalVertexArray[vertIndices.z] = vert2;

      finalVertexAttributeArray[vertIndices.x] = vertAttrib0;
      finalVertexAttributeArray[vertIndices.y] = vertAttrib1;
      finalVertexAttributeArray[vertIndices.z] = vertAttrib2;

      finalAttributeArray[index + primitiveOffset].quadIndex = constructUint4(vertIndices, -1);
    }
    else if (primitiveType == PrimitiveIndexedQuad)
    {
      uint4 vertIndices;
      {
        vertIndices = primitiveAttribPtr[index].quadIndex;
      }

      // get vertex zero and vertex position
      PrimitiveStruct vert0 = primitiveBuffer[vertIndices.x];
      PrimitiveStruct vert1 = primitiveBuffer[vertIndices.y];
      PrimitiveStruct vert2 = primitiveBuffer[vertIndices.z];
      PrimitiveStruct vert3;

      vert0.position = mulMatrixVec(matrix, constructFloat4(vert0.position, 1.f)).xyz;
      vert1.position = mulMatrixVec(matrix, constructFloat4(vert1.position, 1.f)).xyz;
      vert2.position = mulMatrixVec(matrix, constructFloat4(vert2.position, 1.f)).xyz;

      const VertexAttrib vertAttrib0 = vertexAttribPtr[vertIndices.x];
      const VertexAttrib vertAttrib1 = vertexAttribPtr[vertIndices.y];
      const VertexAttrib vertAttrib2 = vertexAttribPtr[vertIndices.z];
      VertexAttrib vertAttrib3;

      if (vertIndices.w != -1)
      {
        vert3 = primitiveBuffer[vertIndices.w];
        vert3.position = mulMatrixVec(matrix, constructFloat4(vert3.position, 1.f)).xyz;
        vertAttrib3 = vertexAttribPtr[vertIndices.w];
        vertIndices.w += vertexOffset;
        vert3.identity = primitiveIdentity;
      }

      if (primitiveType == PrimitiveIndexedQuad)
      {
        vertIndices.xyz += vertexOffset;
      }

      vert0.identity = primitiveIdentity;
      vert1.identity = primitiveIdentity;
      vert2.identity = primitiveIdentity;

      finalVertexArray[vertIndices.x] = vert0;
      finalVertexArray[vertIndices.y] = vert1;
      finalVertexArray[vertIndices.z] = vert2;

      if (vertIndices.w != -1)
      {
        finalVertexArray[vertIndices.w] = vert3;
        finalVertexAttributeArray[vertIndices.w] = vertAttrib3;
      }

      finalVertexAttributeArray[vertIndices.x] = vertAttrib0;
      finalVertexAttributeArray[vertIndices.y] = vertAttrib1;
      finalVertexAttributeArray[vertIndices.z] = vertAttrib2;

      finalAttributeArray[index + primitiveOffset].quadIndex = vertIndices;
    }
  }
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

  // code of this leaf
  const uint2 currNodePrefix = constructUint2(bvhLeafs[internalNodeIndex].mortonCode, internalNodeIndex);

  const short nodeLeftSharedPrefixLength  = getNodesCommonPrefixLength(bvhLeafs, internalNodeIndex - 1, nodeCount, currNodePrefix);
  const short nodeRightSharedPrefixLength = getNodesCommonPrefixLength(bvhLeafs, internalNodeIndex + 1, nodeCount, currNodePrefix);

  const int direction   = select(-1, 1, nodeRightSharedPrefixLength > nodeLeftSharedPrefixLength);
  const short minLength = select(nodeRightSharedPrefixLength, nodeLeftSharedPrefixLength, (short)(nodeRightSharedPrefixLength > nodeLeftSharedPrefixLength));

  int maxStep = 8;
  int currInternalNodeIndex;

  // Compute upper bound for the length of the range
  do
  {
    maxStep <<= 1;
    currInternalNodeIndex = internalNodeIndex + maxStep * direction;
  }
  while (getNodesCommonPrefixLength(bvhLeafs, currInternalNodeIndex, nodeCount, currNodePrefix) > minLength);

  int indexOffset = 0;

  // Find the exact offset using binary search
  for (int currentStep = maxStep >> 1; currentStep != 0; currentStep >>= 1)
  {
    currInternalNodeIndex = internalNodeIndex + (currentStep + indexOffset) * direction;
    if (getNodesCommonPrefixLength(bvhLeafs, currInternalNodeIndex, nodeCount, currNodePrefix) > minLength)
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
    if (getNodesCommonPrefixLength(bvhLeafs, currInternalNodeIndex, nodeCount, currNodePrefix) > splitLength)
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

  if (!leftIsLeaf)  nodeParentNodeIndices[splitOffset]     = setBVHInternalNodeMarker(false, internalNodeIndex);
  if (!rightIsLeaf) nodeParentNodeIndices[splitOffset + 1] = setBVHInternalNodeMarker(false, internalNodeIndex);

  if (leftIsLeaf)   leafParentNodeIndices[splitOffset]     = setBVHInternalNodeMarker(false, internalNodeIndex);
  if (rightIsLeaf)  leafParentNodeIndices[splitOffset + 1] = setBVHInternalNodeMarker(false, internalNodeIndex);

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
    currentNodeIndex = removeBVHInternalNodeMarker(currentNodeIndex);
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
      childBoundingBox = atomicLoadXAB(treeInternalNodeBoundingBoxes + removeBVHInternalNodeMarker(internalNode.childLeft));
    }
    mergeXAB(&mergedBoundingBox, &childBoundingBox);

    if (isBVHLeafNode(internalNode.childRight))
    {
      childBoundingBox = primitiveBoundingBoxes[internalNode.childRight];
    }
    else
    {
      childBoundingBox = atomicLoadXAB(treeInternalNodeBoundingBoxes + removeBVHInternalNodeMarker(internalNode.childRight));
    }
    mergeXAB(&mergedBoundingBox, &childBoundingBox);

    // save the internal node information
    atomicStoreXAB(treeInternalNodeBoundingBoxes + currentNodeIndex, mergedBoundingBox);

    // process the parent node next
    currentNodeIndex = nodeParentNodeIndices[currentNodeIndex];
  }
}

#endif
