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
  HitStruct hit;
  initializeHit(&hit);

  hit.distance = currentTime;

  const uint rootNode    = setBVHInternalNodeMarker(false, 0);

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
      currNodeIndex = rootNode;
      break;
#endif
    }
  }

  return hit;
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
      boundingBox = treeLeafNodeBoundingBoxes[parentNode.child[i]];
    }
    else
    {
      boundingBox = treeInternalNodeBoundingBoxes[removeBVHInternalNodeMarker(parentNode.child[i])];
    }

    const float time = rayXABIntersectTime(timeIn, boundingBox, rayOrigin, invRayDirection, sign);
    if (time != -1.f && minTime > time)
    {
      ret[i] = 1;
      ret[2] = i;
      minTime = time;
    }
  }

  return constructUshort3(ret[0], ret[1], ret[2]);
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
  Shared uint*                  sharedLeafNodeIndex,
  const bool                    isPrimitiveADS = false)
{
  HitStruct hit;
  initializeHit(&hit);

  hit.distance = currentTime;

  const uint rootNode    = setBVHInternalNodeMarker(false, 0);

  ushort traverseState   = BVH_TRAVERSAL_FROM_PARENT;
  BVHNodeInfo parentNode = treeInternalNodes[0];

  const uchar3 signBits  = select(constructUchar3(0), constructUchar3(1), sign);
  // flip near plane if 2 or more negatives are in the ray direction
  // a simple approach to possible get an intersection sooner
  ushort nearPlane       = getNearChilds(treeLeafNodeBoundingBoxes, treeInternalNodeBoundingBoxes, parentNode, rayOrigin, invRayDirection, currentTime, sign).z;
  uint currNodeIndex     = parentNode.child[nearPlane];
  uint parentNodeIndex   = rootNode;

#if RAY_TRAVERSAL_SHARED_MEMORY_INDEX_STRIDE > 0
  Shared uint *leafNodeIndex = sharedLeafNodeIndex;
#else
  uint leafNodeIndex[RAY_TRAVERSAL_BVH_MAX_LEAFS];
#endif

#ifdef PRIMITIVE_INSTANCE_TRAVERSAL
  if (!isPrimitiveADS && leafNodeIndex[localIndex * RAY_TRAVERSAL_SHARED_MEMORY_INDEX_STRIDE + RAY_TRAVERSAL_BVH_MAX_LEAFS] != -1)
  {
    traverseState   = leafNodeIndex[localIndex * RAY_TRAVERSAL_SHARED_MEMORY_INDEX_STRIDE + RAY_TRAVERSAL_BVH_MAX_LEAFS];
    currNodeIndex   = leafNodeIndex[localIndex * RAY_TRAVERSAL_SHARED_MEMORY_INDEX_STRIDE + RAY_TRAVERSAL_BVH_MAX_LEAFS + 1];
    parentNodeIndex = leafNodeIndex[localIndex * RAY_TRAVERSAL_SHARED_MEMORY_INDEX_STRIDE + RAY_TRAVERSAL_BVH_MAX_LEAFS + 2];
    parentNode      = treeInternalNodes[parentNodeIndex];
  }
#endif

  ushort leafCount = 0;

  // main intersection loop
  while (currNodeIndex != rootNode)
  {
    // traverse while a leaf node is found
    while (currNodeIndex != rootNode)
    {
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
          parentNodeIndex = nodeParentNodeIndices[removeBVHInternalNodeMarker(currNodeIndex)];
        }

        // fetch parent node, since it will be available otherwise
        parentNode = treeInternalNodes[removeBVHInternalNodeMarker(parentNodeIndex)];
        nearPlane = getNearChilds(treeLeafNodeBoundingBoxes, treeInternalNodeBoundingBoxes, parentNode, rayOrigin, invRayDirection, currentTime, sign).z;

        // if near is processed
        const bool stateIsLeft = (currNodeIndex == parentNode.child[nearPlane]);
        traverseState = select(BVH_TRAVERSAL_FROM_CHILD, BVH_TRAVERSAL_FROM_SIBLING, stateIsLeft);
        currNodeIndex = select(parentNodeIndex, parentNode.child[nearPlane^1], stateIsLeft);
        continue;
      }

      // when coming from parent or sibling
      const bool stateIsSibling = (traverseState == BVH_TRAVERSAL_FROM_SIBLING);
      traverseState = select(BVH_TRAVERSAL_FROM_SIBLING, BVH_TRAVERSAL_FROM_CHILD, stateIsSibling);

      const bool isLeaf = isBVHLeafNode(currNodeIndex);
      const uint noNodeCurrNodeIndex = removeBVHInternalNodeMarker(currNodeIndex);

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

      // switch to parent or sibling when internal node is not intersecting
      const uint nextNodeIndex = select(parentNode.child[nearPlane^1], parentNodeIndex, stateIsSibling);

      // if current node is leaf, mark for test
      if (isLeaf)
      {
#ifdef PRIMITIVE_INSTANCE_TRAVERSAL
        if (!isPrimitiveADS && leafNodeIndex[localIndex * RAY_TRAVERSAL_SHARED_MEMORY_INDEX_STRIDE + RAY_TRAVERSAL_BVH_MAX_LEAFS] == -1 && intersectsBVH)
        {
          leafNodeIndex[localIndex * RAY_TRAVERSAL_SHARED_MEMORY_INDEX_STRIDE + RAY_TRAVERSAL_BVH_MAX_LEAFS]      = traverseState;
          leafNodeIndex[localIndex * RAY_TRAVERSAL_SHARED_MEMORY_INDEX_STRIDE + RAY_TRAVERSAL_BVH_MAX_LEAFS + 1]  = nextNodeIndex;
          leafNodeIndex[localIndex * RAY_TRAVERSAL_SHARED_MEMORY_INDEX_STRIDE + RAY_TRAVERSAL_BVH_MAX_LEAFS + 2]  = parentNodeIndex;
          leafNodeIndex[localIndex * RAY_TRAVERSAL_SHARED_MEMORY_INDEX_STRIDE + RAY_TRAVERSAL_BVH_MAX_LEAFS + 3]  = currNodeIndex;
          return hit;
        }
#endif
        if (intersectsBVH)
        {
          leafNodeIndex[localIndex * RAY_TRAVERSAL_SHARED_MEMORY_INDEX_STRIDE + leafCount++] = currNodeIndex;
        }
        currNodeIndex = nextNodeIndex;
        if (leafCount != RAY_TRAVERSAL_BVH_MAX_LEAFS)
        {
          continue;
        }
        break;
      }
      else
      {
        // if internal node test bounding box for intersection
        if (intersectsBVH)
        {
          addBVHHit(hit.bvhHits, 1);
        }
        else
        {
          currNodeIndex = nextNodeIndex;
          continue;
        }
      }

      parentNode      = treeInternalNodes[noNodeCurrNodeIndex];
      nearPlane       = getNearChilds(treeLeafNodeBoundingBoxes, treeInternalNodeBoundingBoxes, parentNode, rayOrigin, invRayDirection, currentTime, sign).z;
      parentNodeIndex = currNodeIndex;
      currNodeIndex   = parentNode.child[nearPlane];
      traverseState   = BVH_TRAVERSAL_FROM_PARENT;
    }

#ifdef PRIMITIVE_INSTANCE_TRAVERSAL
    if (isPrimitiveADS)
#endif
    for (ushort i=0; i<leafCount; i++)
    {
      // test colision if not an invalid node
      if (earliestIntersection(&hit, leafNodeIndex[localIndex * RAY_TRAVERSAL_SHARED_MEMORY_INDEX_STRIDE + i], rayOrigin, rayDirection, invRayDirection, sign, vertexArray, attributeArray, systemSettings, primInfo))
      {
#ifdef IntersectionTypeAny
        currNodeIndex = rootNode;
        break;
#endif
      }
    }
    leafCount = 0;
  }

  return hit;
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
  Thread DecodedPrimitiveInfo*  primInfo)
{
  HitStruct hit;
  initializeHit(&hit);

  hit.distance = currentTime;

  short stackTop = 0;
  uint traversalStack[64];

  // mark index of the root node internal
  uint currNodeIndex = setBVHInternalNodeMarker(false, 0);

  while (true)
  {
    // traverse while a leaf node is found
    while (!isBVHLeafNode(currNodeIndex))
    {
      BVHNodeInfo node = treeInternalNodes[removeBVHInternalNodeMarker(currNodeIndex)];
      const XAB leftBoundingBox  = treeInternalNodeBoundingBoxes[removeBVHInternalNodeMarker(node.childLeft)];
      const XAB rightBoundingBox = treeInternalNodeBoundingBoxes[removeBVHInternalNodeMarker(node.childRight)];

      float firstDist  = hit.distance;
      float secondDist = hit.distance;

      bool addFirst  = isBVHLeafNode(node.childLeft)  || rayXABIntersectEarliest(&firstDist, leftBoundingBox, rayOrigin, invRayDirection, sign);
      bool addSecond = isBVHLeafNode(node.childRight) || rayXABIntersectEarliest(&secondDist, rightBoundingBox, rayOrigin, invRayDirection, sign);

      addBVHHit(hit.bvhHits, addFirst);
      addBVHHit(hit.bvhHits, addSecond);

      // swap position and put closer one later so that its picked up first
      if (addFirst && addSecond && firstDist < secondDist)
      {
        bool addTemp    = addFirst;
        addFirst        = addSecond;
        addSecond       = addTemp;

        uint childTemp  = node.childLeft;
        node.childLeft  = node.childRight;
        node.childRight = childTemp;
      }

      if (addFirst)
      {
        traversalStack[stackTop++] = node.childLeft;
      }

      if (addSecond)
      {
        traversalStack[stackTop++] = node.childRight;
      }

      // break if the stack is empty
      if (stackTop == 0)
      {
        // mark node invalid
        currNodeIndex = BOUNDING_VOLUME_HIERARCHY_ADS_ROOT_NODE_MARKER;
        break;
      }

      currNodeIndex = traversalStack[--stackTop];
    }

    // test colision if not an invalid node
    if (currNodeIndex != BOUNDING_VOLUME_HIERARCHY_ADS_ROOT_NODE_MARKER)
    {
      if (earliestIntersection(&hit, currNodeIndex, rayOrigin, rayDirection, invRayDirection, sign, vertexArray, attributeArray, systemSettings, primInfo))
      {
#ifdef IntersectionTypeAny
        break;
#endif
      }
    }

    // exit if nothing to fetch
    if (stackTop == 0)
    {
      break;
    }

    // pop from the stack
    currNodeIndex = traversalStack[--stackTop];
  }

  return hit;
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
  const bool3 sign = selectInput3(invRayDirection < 0.f);

  DecodedPrimitiveInfo primInfo;
  primInfo.primitiveType = RTPrimitiveCount;

  //HitStruct hit = stackTraverseBinaryTree(rays[index].maxDistance, treeInternalNodes, leafParentNodeIndices, nodeParentNodeIndices, treeInternalNodeBoundingBoxes, rayOrigin, rayDirection, invRayDirection, sign, vertexArray, attributeArray, systemSettings);
  HitStruct hit = stacklessTraverseBinaryTree(rays[index].maxDistance, treeInternalNodes, leafParentNodeIndices, nodeParentNodeIndices, treeLeafNodeBoundingBoxes, treeInternalNodeBoundingBoxes, ray.origin, ray.direction, invRayDirection, sign, vertexArray, attributeArray, systemSettings, &primInfo, threadLocalIndex(), sharedLeafNodeIndex);

  traversalSetHitNormal(ray, &hit, vertexArray, attributeArray, vertexAttributeArray, systemSettings);
#ifdef IntersectionTypeClosest
  hits[index] = hit;
#endif
#ifdef IntersectionTypeAny
  setHitPrimitiveIndex(hits[index].primitiveIndex, hit.primitiveIndex);
  setHitPrimitiveIdentity(hits[index].primitiveIdentity, hit.primitiveIdentity);
#endif
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

  DecodedPrimitiveInfo primInfo;
  primInfo.primitiveType = RTPrimitiveCount;

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
    const bool3 sign = selectInput3(invRayDirection < 0.f);

    //HitStruct hit = stackTraverseBinaryTree(rays[index].maxDistance, treeInternalNodes, leafParentNodeIndices, nodeParentNodeIndices, treeInternalNodeBoundingBoxes, rayOrigin, rayDirection, invRayDirection, sign, vertexArray, attributeArray, systemSettings);
    HitStruct hit = stacklessTraverseBinaryTree(rays[index].maxDistance, treeInternalNodes, leafParentNodeIndices, nodeParentNodeIndices, treeLeafNodeBoundingBoxes, treeInternalNodeBoundingBoxes, ray.origin, ray.direction, invRayDirection, sign, vertexArray, attributeArray, systemSettings, &primInfo, threadLocalIndex(), sharedLeafNodeIndex);

    traversalSetHitNormal(ray, &hit, vertexArray, attributeArray, vertexAttributeArray, systemSettings);
#ifdef IntersectionTypeClosest
    hits[index] = hit;
#endif
#ifdef IntersectionTypeAny
    setHitPrimitiveIndex(hits[index].primitiveIndex, hit.primitiveIndex);
    setHitPrimitiveIdentity(hits[index].primitiveIdentity, hit.primitiveIdentity);
#endif
  }
}

#endif

#endif
