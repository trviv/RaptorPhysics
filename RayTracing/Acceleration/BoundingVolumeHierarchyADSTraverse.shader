#ifndef BOUNDING_VOLUME_HIERARCHY_ADS_TRAVERSE_SHADER_H
#define BOUNDING_VOLUME_HIERARCHY_ADS_TRAVERSE_SHADER_H

#define BVH_TRAVERSAL_FROM_PARENT   1
#define BVH_TRAVERSAL_FROM_CHILD    2
#define BVH_TRAVERSAL_FROM_SIBLING  3

short Const int16ComplementaryMasks[16] = {
  ~(1 << 0),  ~(1 << 1),  ~(1 << 2),  ~(1 << 3),  ~(1 << 4),  ~(1 << 5),  ~(1 << 6),  ~(1 << 7),
  ~(1 << 8),  ~(1 << 9),  ~(1 << 10), ~(1 << 11), ~(1 << 12), ~(1 << 13), ~(1 << 14), (short)~(1 << 15)
};

inline void setChildArrayBit(Thread ushort *nearChild, short nearChildIndex, const bool nearChildBit)
{
  const short offset   = nearChildIndex >> 4;
  const short bitIndex = nearChildIndex & 15;
  nearChild[offset]    = (nearChild[offset] & ~(1 << bitIndex)) | ((nearChildBit & 1) << bitIndex);
  //nearChild[offset]    = (nearChild[offset] & int16ComplementaryMasks[bitIndex]) | (nearChildBit << bitIndex);
}

inline ushort getNearChildIndex(Thread ushort *nearChild, const short nearChildIndex)
{
  const short offset   = nearChildIndex >> 4;
  const short bitIndex = nearChildIndex & 15;
  return (nearChild[offset] >> bitIndex) & 1;
}

inline BVHNodeInfo fetchInternalNode(
  Thread ushort*            nearChild,
  Thread ushort*            nearChildXABTest,
  Thread ushort*            farChildXABTest,
  Thread short*             nearChildIndex,
  const Device BVHNodeInfo* treeInternalNodes,
  const Device XAB*         treeInternalNodeBoundingBoxes,
  const float3              rayOrigin,
  const float3              rayDirection,
  const float3              invRayDirection,
  const bool3               sign,
  const float               currentTime,
  const uint                nodeIndex)
{
  const BVHNodeInfo node     = treeInternalNodes[removeBVHInternalNodeMarker(nodeIndex)];
  const XAB leftBoundingBox  = treeInternalNodeBoundingBoxes[removeBVHInternalNodeMarker(node.childLeft)];
  const XAB rightBoundingBox = treeInternalNodeBoundingBoxes[removeBVHInternalNodeMarker(node.childRight)];

  float firstDist  = currentTime;
  float secondDist = currentTime;

  (*nearChildIndex)++;

  bool firstTest  = rayXABIntersectEarliest(&firstDist,  leftBoundingBox,  rayOrigin, invRayDirection, sign);
  bool secondTest = rayXABIntersectEarliest(&secondDist, rightBoundingBox, rayOrigin, invRayDirection, sign);

  const bool swapChilds = firstDist > secondDist;

  if (swapChilds)
  {
    bool temp  = firstTest;
    firstTest  = secondTest;
    secondTest = temp;
  }

  setChildArrayBit(nearChild,        *nearChildIndex, swapChilds);
  setChildArrayBit(nearChildXABTest, *nearChildIndex, firstTest);
  setChildArrayBit(farChildXABTest,  *nearChildIndex, secondTest);

  return node;
}

inline HitStruct stacklessTraverseBinaryTree2(
  float                         currentTime,
  const Device BVHNodeInfo*     treeInternalNodes,
  const Device uint*            leafParentNodeIndices,
  const Device uint*            nodeParentNodeIndices,
  const Device XAB*             treeLeafNodeBoundingBoxes,
  const Device XAB*             treeInternalNodeBoundingBoxes,
  const float3                  rayOrigin,
  const float3                  rayDirection,
  const float3                  invRayDirection,
  const bool3                   sign,
  const Device PrimitiveStruct* vertexArray,
  const Device PrimitiveAttrib* attributeArray,
  Const RTSystemSettings*       systemSettings,
  Thread DecodedPrimitiveInfo*  primInfo)
{
  HitStruct hit       = defaultHit(currentTime);
  const uint rootNode = setBVHInternalNodeMarker(false, 0);

  ushort nearChild[3];
  short  nearChildIndex   = -1;

  ushort nearChildXABTest[3];
  ushort farChildXABTest[3];

  ushort traverseState   = BVH_TRAVERSAL_FROM_PARENT;

  BVHNodeInfo parentNode = fetchInternalNode(nearChild, nearChildXABTest, farChildXABTest, &nearChildIndex, treeInternalNodes, treeInternalNodeBoundingBoxes,
    rayOrigin, rayDirection, invRayDirection, sign, currentTime, 0);

  uint currNodeIndex     = parentNode.child[getNearChildIndex(nearChild, nearChildIndex)];
  uint parentNodeIndex   = rootNode;

  // main intersection loop
  while (currNodeIndex != rootNode)
  {
    uint leafNodeIndex;

    // traverse while a leaf node is found
    while (currNodeIndex != rootNode)
    {
      // when going to parent from child
      if (traverseState == BVH_TRAVERSAL_FROM_CHILD)
      {
        nearChildIndex--;
        // fetch parent index
        if (isBVHLeafNode(currNodeIndex))
        {
          parentNodeIndex = leafParentNodeIndices[currNodeIndex];
        }
        else
        {
          parentNodeIndex = nodeParentNodeIndices[removeBVHInternalNodeMarker(currNodeIndex)];
        }

        // fetch parent node, since it will be available otherwise
        parentNode = treeInternalNodes[removeBVHInternalNodeMarker(parentNodeIndex)];

        // if near is processed
        const bool stateIsLeft = (currNodeIndex == parentNode.child[getNearChildIndex(nearChild, nearChildIndex)]);
        traverseState = select(BVH_TRAVERSAL_FROM_CHILD, BVH_TRAVERSAL_FROM_SIBLING, stateIsLeft);
        currNodeIndex = select(parentNodeIndex, parentNode.child[1^getNearChildIndex(nearChild, nearChildIndex)], stateIsLeft);
        continue;
      }

      // when coming from parent or sibling
      const bool stateIsSibling = (traverseState == BVH_TRAVERSAL_FROM_SIBLING);
      traverseState = select(BVH_TRAVERSAL_FROM_SIBLING, BVH_TRAVERSAL_FROM_CHILD, stateIsSibling);

      // if current node is leaf, mark for test
      if (isBVHLeafNode(currNodeIndex))
      {
        leafNodeIndex = currNodeIndex;
        currNodeIndex = select(parentNode.child[1^getNearChildIndex(nearChild, nearChildIndex)], parentNodeIndex, stateIsSibling);
        break;
      }

      Thread ushort *childXABTestArray;
      if (stateIsSibling)
      {
        childXABTestArray = farChildXABTest;
      }
      else
      {
        childXABTestArray = nearChildXABTest;
      }
      // if internal node test bounding box for intersection
      //const XAB boundingBox = treeInternalNodeBoundingBoxes[removeBVHInternalNodeMarker(currNodeIndex)];
      //if (!rayXABIntersectTest(hit.distance, boundingBox, rayOrigin, invRayDirection, sign))
      if (getNearChildIndex(childXABTestArray, nearChildIndex) == 0)
      {
        // switch to parent or sibling when internal node is not intersecting
        currNodeIndex = select(parentNode.child[1^getNearChildIndex(nearChild, nearChildIndex)], parentNodeIndex, stateIsSibling);
        continue;
      }

      //parentNode      = treeInternalNodes[removeBVHInternalNodeMarker(currNodeIndex)];
      parentNode      = fetchInternalNode(nearChild, nearChildXABTest, farChildXABTest, &nearChildIndex, treeInternalNodes, treeInternalNodeBoundingBoxes,
        rayOrigin, rayDirection, invRayDirection, sign, currentTime, currNodeIndex);
      parentNodeIndex = currNodeIndex;
      currNodeIndex   = parentNode.child[getNearChildIndex(nearChild, nearChildIndex)];
      traverseState   = BVH_TRAVERSAL_FROM_PARENT;
    }

    // test colision if not an invalid node
    if (earliestIntersection(&hit, leafNodeIndex, rayOrigin, rayDirection, invRayDirection, sign, vertexArray, attributeArray, systemSettings, primInfo))
    {
#ifdef IntersectionTypeAny
      return hit;
#endif
    }
  }

  return hit;
}

inline uint getChildNode(const BVHNodeInfo parentNode, const ushort child)
{
  return select(parentNode.childLeft, parentNode.childRight, child);
}

inline ushort3 getNearChilds(
  const Device XAB* treeLeafNodeBoundingBoxes,
  const Device XAB* treeInternalNodeBoundingBoxes,
  const BVHNodeInfo parentNode,
  const float3      rayOrigin,
  const float3      invRayDirection,
  const float       timeIn,
  const bool3       sign)
{
#ifdef STACKLESS_TRAVERSE_EARLY_CHILD
  XAB boundingBox;
  ushort ret[3] = {0, 0, 0};
  float minTime = INFINITY;

#pragma unroll
  for (short i=0; i<2; i++)
  {
    if (isBVHLeafNode(parentNode.child[i]))
    {
      boundingBox = treeLeafNodeBoundingBoxes[getChildNode(parentNode, i)];
    }
    else
    {
      boundingBox = treeInternalNodeBoundingBoxes[removeBVHInternalNodeMarker(getChildNode(parentNode, i))];
    }

    const float time = rayXABIntersectTime(timeIn, boundingBox, rayOrigin, invRayDirection, sign);
    ret[i] = (time != INFINITY);
    if (minTime > time)
    {
      ret[2] = i;
      minTime = time;
    }
  }

  return constructUshort3(ret[0], ret[1], ret[2]);
#else
  return constructUshort3(0, 0, 0);
#endif
}

inline void setNearChildArrayBit(Thread uint *nearChildBitArray, Thread uint *farChildArray,
  const ushort stackBitOffset, const ushort3 nearChilds)
{
#ifdef STACKLESS_TRAVERSE_EARLY_CHILD
  const short offset    = stackBitOffset >> 5;
  const short bitIndex  = stackBitOffset & 31;
  const uint mask       = ~(1 << bitIndex);

  nearChildBitArray[offset] = (nearChildBitArray[offset] & mask) | ((nearChilds.z & 1) << bitIndex);
  farChildArray[offset]     = (farChildArray[offset]     & mask) | ((nearChilds.y & 1) << bitIndex);
#endif
}

inline ushort3 getNearChildArrayBit(Thread uint *nearChildBitArray, Thread uint *farChildArray,
  const ushort stackBitOffset)
{
#ifdef STACKLESS_TRAVERSE_EARLY_CHILD
  const short offset        = stackBitOffset >> 5;
  const short bitIndex      = stackBitOffset & 31;

  ushort3 ret;
  ret.z = (nearChildBitArray[offset] >> bitIndex);
  ret.y = (farChildArray[offset]     >> bitIndex);

  return ret & 1;
#else
  return constructUshort3(0, 0, 0);
#endif
}

inline HitStruct stacklessTraverseBinaryTree(
  float                         currentTime,
  const Device BVHNodeInfo*     treeInternalNodes,
  const Device uint*            leafParentNodeIndices,
  const Device uint*            nodeParentNodeIndices,
  const Device XAB*             treeLeafNodeBoundingBoxes,
  const Device XAB*             treeInternalNodeBoundingBoxes,
  const float3                  rayOrigin,
  const float3                  rayDirection,
  const float3                  invRayDirection,
  const bool3                   sign,
  const Device PrimitiveStruct* vertexArray,
  const Device PrimitiveAttrib* attributeArray,
  Const RTSystemSettings*       systemSettings,
  Thread DecodedPrimitiveInfo*  primInfo,
  const ushort                  localIndex,
  Shared uint*                  sharedLeafNodeIndex)
{
  HitStruct hit          = defaultHit(currentTime);
  const uint rootNode    = setBVHInternalNodeMarker(false, 0);
  ushort traverseState   = BVH_TRAVERSAL_FROM_PARENT;
  BVHNodeInfo parentNode = treeInternalNodes[0];

  const uchar3 signBits  = select(constructUchar3(0), constructUchar3(1), sign);
  // flip near plane if 2 or more negatives are in the ray direction
  // a simple approach to possible get an intersection sooner
  ushort3 nearChilds     = getNearChilds(treeLeafNodeBoundingBoxes, treeInternalNodeBoundingBoxes, parentNode, rayOrigin, invRayDirection, currentTime, sign);
  uint currNodeIndex     = getChildNode(parentNode, nearChilds.z);
  uint parentNodeIndex   = rootNode;

  Thread uint nearChildBitArray[2];
  Thread uint farChildArray[2];
  ushort stackBitOffset = 0;

  setNearChildArrayBit(nearChildBitArray, farChildArray, stackBitOffset, nearChilds);

#if RAY_TRAVERSAL_SHARED_MEMORY_INDEX_STRIDE > 0
  Shared uint *leafNodeIndex = sharedLeafNodeIndex;
#elif RAY_TRAVERSAL_BVH_MAX_LEAFS > 0
  uint leafNodeIndex[RAY_TRAVERSAL_BVH_MAX_LEAFS];
#endif

  ushort leafCount = 0;

  // main intersection loop
  while (currNodeIndex != rootNode)
  {
    // traverse while a leaf node is found
    while (currNodeIndex != rootNode)
    {
      const uint noNodeCurrNodeIndex = removeBVHInternalNodeMarker(currNodeIndex);

      // when going to parent from child
      if (traverseState == BVH_TRAVERSAL_FROM_CHILD)
      {
        // fetch parent index
        if (isBVHLeafNode(currNodeIndex))
        {
          parentNodeIndex = leafParentNodeIndices[currNodeIndex];
        }
        else
        {
          parentNodeIndex = nodeParentNodeIndices[noNodeCurrNodeIndex];
        }

        // fetch parent node, since it will be available otherwise
        parentNode = treeInternalNodes[removeBVHInternalNodeMarker(parentNodeIndex)];
        stackBitOffset--;
        nearChilds = getNearChildArrayBit(nearChildBitArray, farChildArray, stackBitOffset);

        // if near is processed
        const bool stateIsLeft = (currNodeIndex == getChildNode(parentNode, nearChilds.z));
        traverseState = select(BVH_TRAVERSAL_FROM_CHILD, BVH_TRAVERSAL_FROM_SIBLING, stateIsLeft);
        currNodeIndex = select(parentNodeIndex, getChildNode(parentNode, nearChilds.z^1), stateIsLeft);
        continue;
      }

      // when coming from parent or sibling
      const bool stateIsSibling = (traverseState == BVH_TRAVERSAL_FROM_SIBLING);
      traverseState = select(BVH_TRAVERSAL_FROM_SIBLING, BVH_TRAVERSAL_FROM_CHILD, stateIsSibling);

      const bool isLeaf = isBVHLeafNode(currNodeIndex);

#ifndef STACKLESS_TRAVERSE_EARLY_CHILD
      XAB boundingBox;
      if (isLeaf)
      {
        boundingBox = treeLeafNodeBoundingBoxes[currNodeIndex];
      }
      else
      {
        boundingBox = treeInternalNodeBoundingBoxes[noNodeCurrNodeIndex];
      }

      const bool intersectsBVH = rayXABIntersectTest(hit.distance, boundingBox, rayOrigin, invRayDirection, sign);
#else
      bool intersectsBVH;
      if (!stateIsSibling)
      {
        intersectsBVH = select(nearChilds.x, nearChilds.y, nearChilds.z);
      }
      else
      {
        nearChilds    = getNearChildArrayBit(nearChildBitArray, farChildArray, stackBitOffset);
        intersectsBVH = nearChilds.y;
      }
#endif

      // switch to parent or sibling when internal node is not intersecting
      const uint nextNodeIndex = select(getChildNode(parentNode, nearChilds.z^1), parentNodeIndex, stateIsSibling);

      if (intersectsBVH)
      {
        // if leaf mark for test
        if (isLeaf)
        {
#if RAY_TRAVERSAL_BVH_MAX_LEAFS == 0
          if (earliestIntersection(&hit, currNodeIndex, rayOrigin, rayDirection, invRayDirection, sign, vertexArray, attributeArray, systemSettings, primInfo))
          {
#ifdef IntersectionTypeAny
            return hit;
#endif
          }
#else
          leafNodeIndex[localIndex * RAY_TRAVERSAL_SHARED_MEMORY_INDEX_STRIDE + leafCount++] = currNodeIndex;
#endif
        }
        // if internal node move to the node
        else
        {
          addBVHHit(hit.bvhHits, 1);

          parentNode      = treeInternalNodes[noNodeCurrNodeIndex];
          nearChilds       = getNearChilds(treeLeafNodeBoundingBoxes, treeInternalNodeBoundingBoxes, parentNode, rayOrigin, invRayDirection, currentTime, sign);
          parentNodeIndex = currNodeIndex;
          currNodeIndex   = getChildNode(parentNode, nearChilds.z);
          traverseState   = BVH_TRAVERSAL_FROM_PARENT;

          stackBitOffset++;
          setNearChildArrayBit(nearChildBitArray, farChildArray, stackBitOffset, nearChilds);

          continue;
        }
      }

      currNodeIndex = nextNodeIndex;
      if (leafCount == RAY_TRAVERSAL_BVH_MAX_LEAFS)
      {
        break;
      }
    }

#if RAY_TRAVERSAL_BVH_MAX_LEAFS > 0
    for (ushort i=0; i<leafCount; i++)
    {
      // test colision if not an invalid node
      if (earliestIntersection(&hit, leafNodeIndex[localIndex * RAY_TRAVERSAL_SHARED_MEMORY_INDEX_STRIDE + i], rayOrigin, rayDirection, invRayDirection, sign, vertexArray, attributeArray, systemSettings, primInfo))
      {
#ifdef IntersectionTypeAny
        return hit;
#endif
      }
    }
    leafCount = 0;
#endif
  }

  return hit;
}

inline HitStruct stackTraverseBinaryTreeWithInputs(
  Thread HitStruct*             hit,
  const Device BVHNodeInfo*     treeInternalNodes,
  const Device uint*            leafParentNodeIndices,
  const Device uint*            nodeParentNodeIndices,
  const Device XAB*             treeLeafNodeBoundingBoxes,
  const Device XAB*             treeInternalNodeBoundingBoxes,
  const float3                  rayOrigin,
  const float3                  rayDirection,
  const float3                  invRayDirection,
  const bool3                   sign,
  const Device PrimitiveStruct* vertexArray,
  const Device PrimitiveAttrib* attributeArray,
  Const RTSystemSettings*       systemSettings,
  Thread DecodedPrimitiveInfo*  primInfo,
  const ushort                  localIndex,
  Shared uint*                  sharedLeafNodeIndex,
  const short                   stackBase,
  Thread uint*                  traversalStack)
{
  short stackTop = stackBase+1;

  traversalStack[stackBase] = setBVHInternalNodeMarker(false, 0);

  // break if the stack is empty
  while (stackTop > stackBase)
  {
    uint currNodeIndex = traversalStack[--stackTop];

    if (isBVHLeafNode(currNodeIndex))
    {
      if (earliestIntersection(hit, currNodeIndex, rayOrigin, rayDirection, invRayDirection, sign, vertexArray, attributeArray, systemSettings, primInfo))
      {
#ifdef IntersectionTypeAny
        stackTop = stackBase;
#endif
      }
      continue;
    }

    // traverse while a leaf node is found
    const BVHNodeInfo node = treeInternalNodes[removeBVHInternalNodeMarker(currNodeIndex)];

    XAB leftBoundingBox;
    if (isBVHLeafNode(node.childLeft))  leftBoundingBox = treeLeafNodeBoundingBoxes[node.childLeft];
    else                                leftBoundingBox = treeInternalNodeBoundingBoxes[removeBVHInternalNodeMarker(node.childLeft)];

    float leftDist = hit->distance;
    const bool addNear = rayXABIntersectEarliest(&leftDist, leftBoundingBox, rayOrigin, invRayDirection, sign);
    addBVHHit(hit->bvhHits, addNear);

    XAB rightBoundingBox;
    if (isBVHLeafNode(node.childRight)) rightBoundingBox = treeLeafNodeBoundingBoxes[node.childRight];
    else                                rightBoundingBox = treeInternalNodeBoundingBoxes[removeBVHInternalNodeMarker(node.childRight)];

    float rightDist = hit->distance;
    const bool addFar = rayXABIntersectEarliest(&rightDist, rightBoundingBox, rayOrigin, invRayDirection, sign);
    addBVHHit(hit->bvhHits, addFar);

    const bool swapChilds = addNear && addFar && leftDist < rightDist;

    if (addNear) traversalStack[stackTop++] = getChildNode(node, swapChilds);
    if (addFar)  traversalStack[stackTop++] = getChildNode(node, !swapChilds);
  }

  return *hit;
}

inline HitStruct stackTraverseBinaryTree(
  float                         currentTime,
  const Device BVHNodeInfo*     treeInternalNodes,
  const Device uint*            leafParentNodeIndices,
  const Device uint*            nodeParentNodeIndices,
  const Device XAB*             treeLeafNodeBoundingBoxes,
  const Device XAB*             treeInternalNodeBoundingBoxes,
  const float3                  rayOrigin,
  const float3                  rayDirection,
  const float3                  invRayDirection,
  const bool3                   sign,
  const Device PrimitiveStruct* vertexArray,
  const Device PrimitiveAttrib* attributeArray,
  Const RTSystemSettings*       systemSettings,
  Thread DecodedPrimitiveInfo*  primInfo,
  const ushort                  localIndex,
  Shared uint*                  sharedLeafNodeIndex)
{
  uint traversalStack[48];
  HitStruct hit = defaultHit(currentTime);

  return stackTraverseBinaryTreeWithInputs(&hit, treeInternalNodes, leafParentNodeIndices, nodeParentNodeIndices, treeLeafNodeBoundingBoxes, treeInternalNodeBoundingBoxes,
    rayOrigin, rayDirection, invRayDirection, sign, vertexArray, attributeArray, systemSettings, primInfo, localIndex, sharedLeafNodeIndex, 0, traversalStack);
}

#if BVH_ADS_PERSISTENT_MULTIPLIER == 1

Kernel void intersectRaysBVH(
  Device HitStruct*             hits,
  const Device RayStruct*       rays,
  constantKernelInput(uint,     rayCount),
  const Device PrimitiveStruct* vertexArray,
  const Device PrimitiveAttrib* attributeArray,
  const Device VertexAttrib*    vertexAttributeArray,
  const Device BVHNodeInfo*     treeInternalNodes,
  const Device uint*            leafParentNodeIndices,
  const Device uint*            nodeParentNodeIndices,
  const Device XAB*             treeLeafNodeBoundingBoxes,
  const Device XAB*             treeInternalNodeBoundingBoxes,
  Const RTSystemSettings*       systemSettings,
  constantKernelInput(uint,     primitiveCount),
  atomicKernelInput(uint,       rayIndexAtomicBuffer),
  sharedMemKernelInput(uint,    sharedLeafNodeIndex, 14)
  KERNEL_THREAD_ARGUMENTS
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint index = threadIndex();

  if (index >= rayCount)
    return;

  const RayStruct ray = rays[index];
  const float3 invRayDirection = 1.f / ray.direction;
  const bool3 sign = selectInput3(ray.direction < 0.f);

  DecodedPrimitiveInfo primInfo = defaultPrimitiveInfo();

  //HitStruct hit = stackTraverseBinaryTree(rays[index].maxDistance, treeInternalNodes, leafParentNodeIndices, nodeParentNodeIndices, treeInternalNodeBoundingBoxes, rayOrigin, rayDirection, invRayDirection, sign, vertexArray, attributeArray, systemSettings);
  HitStruct hit = stacklessTraverseBinaryTree(rays[index].maxDistance, treeInternalNodes, leafParentNodeIndices, nodeParentNodeIndices, treeLeafNodeBoundingBoxes, treeInternalNodeBoundingBoxes, ray.origin, ray.direction, invRayDirection, sign, vertexArray, attributeArray, systemSettings, &primInfo, threadLocalIndex(), sharedLeafNodeIndex);

  traversalSetHitNormal(ray, &hit, vertexArray, attributeArray, vertexAttributeArray, systemSettings);
  traversalStoreHit(&hits[index], hit);
}

#else

Kernel void intersectRaysBVH(
  Device HitStruct*             hits,
  const Device RayStruct*       rays,
  constantKernelInput(uint,     rayCount),
  const Device PrimitiveStruct* vertexArray,
  const Device PrimitiveAttrib* attributeArray,
  const Device VertexAttrib*    vertexAttributeArray,
  const Device BVHNodeInfo*     treeInternalNodes,
  const Device uint*            leafParentNodeIndices,
  const Device uint*            nodeParentNodeIndices,
  const Device XAB*             treeLeafNodeBoundingBoxes,
  const Device XAB*             treeInternalNodeBoundingBoxes,
  Const RTSystemSettings*       systemSettings,
  constantKernelInput(uint,     primitiveCount),
  atomicKernelInput(uint,       rayIndexAtomicBuffer),
  sharedMemKernelInput(uint,    sharedLeafNodeIndex, 14)
  KERNEL_THREAD_ARGUMENTS)
{
  volatile Shared uint nextRayArray[33];
  volatile Shared uint rayCountArray[33];

  const ushort simdLocalIndex = threadLocalIndex() & (ComputeSimdWidth - 1);
  const ushort simdGroupIndex = threadLocalIndex() >> ComputeSimdWidthExp;

  if (simdLocalIndex == 0)
  {
    rayCountArray[simdGroupIndex] = 0;
  }

  DecodedPrimitiveInfo primInfo = defaultPrimitiveInfo();

  while (true)
  {
    // get rays from global to local pool
    if (rayCountArray[simdGroupIndex] == 0 && simdLocalIndex == 0)
    {
      nextRayArray[simdGroupIndex]  = atomicAdd((rayIndexAtomicBuffer+3), BVH_ADS_PERSISTENT_MULTIPLIER*ComputeSimdWidth);
      rayCountArray[simdGroupIndex] = BVH_ADS_PERSISTENT_MULTIPLIER*ComputeSimdWidth;
    }

    // get rays from local pool
    const uint index = nextRayArray[simdGroupIndex] + simdLocalIndex;
    if (index >= rayCount)
    {
      return;
    }

    if (simdLocalIndex == 0)
    {
      nextRayArray[simdGroupIndex]  += ComputeSimdWidth;
      rayCountArray[simdGroupIndex] -= ComputeSimdWidth;
    }

    const RayStruct ray = rays[index];
    const float3 invRayDirection = 1.f / ray.direction;
    const bool3 sign = selectInput3(ray.direction < 0.f);

    //HitStruct hit = stackTraverseBinaryTree(rays[index].maxDistance, treeInternalNodes, leafParentNodeIndices, nodeParentNodeIndices, treeInternalNodeBoundingBoxes, rayOrigin, rayDirection, invRayDirection, sign, vertexArray, attributeArray, systemSettings);
    HitStruct hit = stacklessTraverseBinaryTree(rays[index].maxDistance, treeInternalNodes, leafParentNodeIndices, nodeParentNodeIndices, treeLeafNodeBoundingBoxes, treeInternalNodeBoundingBoxes, ray.origin, ray.direction, invRayDirection, sign, vertexArray, attributeArray, systemSettings, &primInfo, threadLocalIndex(), sharedLeafNodeIndex);

    traversalSetHitNormal(ray, &hit, vertexArray, attributeArray, vertexAttributeArray, systemSettings);
    traversalStoreHit(&hits[index], hit);
  }
}

#endif

#endif
