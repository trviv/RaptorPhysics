#ifndef BOUNDING_VOLUME_HIERARCHY_ADS_TRAVERSE_SHADER_H
#define BOUNDING_VOLUME_HIERARCHY_ADS_TRAVERSE_SHADER_H

#define BVH_TRAVERSAL_FROM_PARENT   1
#define BVH_TRAVERSAL_FROM_CHILD    2
#define BVH_TRAVERSAL_FROM_SIBLING  3

#define BVH_MAX_LEAFS 4

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
#endif
    }
  }

  return hit;
}

inline HitStruct stacklessTraverseBinaryTree(
  float                         currentTime,
  const Device BVHNodeInfo*     treeInternalNodes,
  const Device uint*            leafParentNodeIndices,
  const Device uint*            nodeParentNodeIndices,
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

  ushort traverseState   = BVH_TRAVERSAL_FROM_PARENT;
  BVHNodeInfo parentNode = treeInternalNodes[0];

  const uchar3 signBits  = select(constructUchar3(0), constructUchar3(1), sign);
  // flip near plane if 2 or more negatives are in the ray direction
  // a simple approach to possible get an intersection sooner
  const uchar nearPlane = 0;//(signBits.x + signBits.y + signBits.z) > 1;
  uint currNodeIndex     = parentNode.child[nearPlane];
  uint parentNodeIndex   = rootNode;

  uchar leafCount = 0;
  uint leafNodeIndex[BVH_MAX_LEAFS];

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

        // if near is processed
        const bool stateIsLeft = (currNodeIndex == parentNode.child[nearPlane]);
        traverseState = select(BVH_TRAVERSAL_FROM_CHILD, BVH_TRAVERSAL_FROM_SIBLING, stateIsLeft);
        currNodeIndex = select(parentNodeIndex, parentNode.child[nearPlane^1], stateIsLeft);
        continue;
      }

      // when coming from parent or sibling
      const bool stateIsSibling = (traverseState == BVH_TRAVERSAL_FROM_SIBLING);
      traverseState = select(BVH_TRAVERSAL_FROM_SIBLING, BVH_TRAVERSAL_FROM_CHILD, stateIsSibling);

      // if current node is leaf, mark for test
      if (isBVHLeafNode(currNodeIndex))
      {
        leafNodeIndex[leafCount++] = currNodeIndex;
        currNodeIndex = select(parentNode.child[nearPlane^1], parentNodeIndex, stateIsSibling);
        if (leafCount != BVH_MAX_LEAFS)
        {
          continue;
        }
        break;
      }

      const uint noNodeCurrNodeIndex = removeBVHInternalNodeMarker(currNodeIndex);

      // if internal node test bounding box for intersection
      const XAB boundingBox = treeInternalNodeBoundingBoxes[noNodeCurrNodeIndex];
      if (!rayXABIntersectTest(hit.distance, boundingBox, rayOrigin, invRayDirection, sign))
      {
        // switch to parent or sibling when internal node is not intersecting
        currNodeIndex = select(parentNode.child[nearPlane^1], parentNodeIndex, stateIsSibling);
        continue;
      }

      parentNode      = treeInternalNodes[noNodeCurrNodeIndex];
      parentNodeIndex = currNodeIndex;
      currNodeIndex   = parentNode.child[nearPlane];
      traverseState   = BVH_TRAVERSAL_FROM_PARENT;
    }

    for (ushort i=0; i<leafCount; i++)
    {
      // test colision if not an invalid node
      if (earliestIntersection(&hit, leafNodeIndex[i], rayOrigin, rayDirection, invRayDirection, sign, vertexArray, attributeArray, systemSettings, primInfo))
      {
#ifdef IntersectionTypeAny
        currNodeIndex = rootNode;
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
  const Device BVHNodeInfo*     treeInternalNodes,
  const Device uint*            leafParentNodeIndices,
  const Device uint*            nodeParentNodeIndices,
  const Device XAB*             treeInternalNodeBoundingBoxes,
  Const RTSystemSettings*       systemSettings,
  constantKernelInput(uint,     primitiveCount)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint index = threadIndex();

  if (index >= rayCount)
    return;

  const float3 rayOrigin = rays[index].origin;
  const float3 rayDirection = rays[index].direction;
  const float3 invRayDirection = 1.f / rayDirection;
  const bool3 sign = selectInput3(invRayDirection < 0.f);

  DecodedPrimitiveInfo primInfo;
  primInfo.primType = RTPrimitiveCount;

  //HitStruct hit = stackTraverseBinaryTree(rays[index].maxDistance, treeInternalNodes, leafParentNodeIndices, nodeParentNodeIndices, treeInternalNodeBoundingBoxes, rayOrigin, rayDirection, invRayDirection, sign, vertexArray, attributeArray, systemSettings);
  HitStruct hit = stacklessTraverseBinaryTree(rays[index].maxDistance, treeInternalNodes, leafParentNodeIndices, nodeParentNodeIndices, treeInternalNodeBoundingBoxes, rayOrigin, rayDirection, invRayDirection, sign, vertexArray, attributeArray, systemSettings, &primInfo);

#ifdef IntersectionTypeClosest
  setHitNormal(hit.normal, select(normalize(hit.normal), 0.f, hit.primitiveIndex == -1));
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
  const Device BVHNodeInfo*     treeInternalNodes,
  const Device uint*            leafParentNodeIndices,
  const Device uint*            nodeParentNodeIndices,
  const Device XAB*             treeInternalNodeBoundingBoxes,
  Const RTSystemSettings*       systemSettings,
  constantKernelInput(uint,     primitiveCount),
  atomicKernelInput(uint,       rayIndexAtomicBuffer)
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
  primInfo.primType = RTPrimitiveCount;

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

    const float3 rayOrigin = rays[index].origin;
    const float3 rayDirection = rays[index].direction;
    const float3 invRayDirection = 1.f / rayDirection;
    const bool3 sign = selectInput3(invRayDirection < 0.f);

    //HitStruct hit = stackTraverseBinaryTree(rays[index].maxDistance, treeInternalNodes, leafParentNodeIndices, nodeParentNodeIndices, treeInternalNodeBoundingBoxes, rayOrigin, rayDirection, invRayDirection, sign, vertexArray, attributeArray, systemSettings);
    HitStruct hit = stacklessTraverseBinaryTree(rays[index].maxDistance, treeInternalNodes, leafParentNodeIndices, nodeParentNodeIndices, treeInternalNodeBoundingBoxes, rayOrigin, rayDirection, invRayDirection, sign, vertexArray, attributeArray, systemSettings, &primInfo);

#ifdef IntersectionTypeClosest
    setHitNormal(hit.normal, select(normalize(hit.normal), 0.f, hit.primitiveIndex == -1));
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
