#ifndef BOUNDING_VOLUME_HIERARCHY_STACKLESS_TRAVERSE_SHADER_H
#define BOUNDING_VOLUME_HIERARCHY_STACKLESS_TRAVERSE_SHADER_H

#include "AccelerationDataStructTraverse.shader"
#include "BoundingVolumeHierarchyADSCreate.shader"

#define BVH_TRAVERSAL_FROM_PARENT   1
#define BVH_TRAVERSAL_FROM_CHILD    2
#define BVH_TRAVERSAL_FROM_SIBLING  3

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

  //const uchar3 signBits  = select(constructUchar3(0), constructUchar3(1), sign);
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

#endif
