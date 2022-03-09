#ifndef BOUNDING_VOLUME_HIERARCHY_STACK_TRAVERSE_SHADER_H
#define BOUNDING_VOLUME_HIERARCHY_STACK_TRAVERSE_SHADER_H

struct BVHNodeIntersectionData
{
  BVHNodeInfo node;
  bool addNear;
  bool addFar;
};

inline BVHNodeIntersectionData getNodeIntersectionData(
  Thread HitStruct*         hit,
  const float3              rayOrigin,
  const float3              invRayDirection,
  const bool3               sign,
  const Device BVHNodeInfo* treeInternalNodes,
  const Device uint*        leafParentNodeIndices,
  const Device uint*        nodeParentNodeIndices,
  const Device XAB*         treeLeafNodeBoundingBoxes,
  const Device XAB*         treeInternalNodeBoundingBoxes,
  const uint                currNodeIndex)
{
  // get the tree node
  BVHNodeInfo node = treeInternalNodes[removeBVHInternalNodeMarker(currNodeIndex)];

  // read the left child
  XAB leftBoundingBox;
  if (isBVHLeafNode(node.childLeft))  leftBoundingBox = treeLeafNodeBoundingBoxes[node.childLeft];
  else                                leftBoundingBox = treeInternalNodeBoundingBoxes[removeBVHInternalNodeMarker(node.childLeft)];

  float leftDist = hit->distance;
  bool addNear = rayXABIntersectEarliest(&leftDist, leftBoundingBox, rayOrigin, invRayDirection, sign);
  addBVHHit(hit->bvhHits, addNear);

  // read the right child
  XAB rightBoundingBox;
  if (isBVHLeafNode(node.childRight)) rightBoundingBox = treeLeafNodeBoundingBoxes[node.childRight];
  else                                rightBoundingBox = treeInternalNodeBoundingBoxes[removeBVHInternalNodeMarker(node.childRight)];

  float rightDist = hit->distance;
  bool addFar = rayXABIntersectEarliest(&rightDist, rightBoundingBox, rayOrigin, invRayDirection, sign);
  addBVHHit(hit->bvhHits, addFar);

  // if both childs are intersecting
  if (addNear && addFar)
  {
    // swap if the left child indices is farther than the right child
    if (leftDist > rightDist)
    {
      const uint temp = node.childLeft;
      node.childLeft  = node.childRight;
      node.childRight = temp;
    }
  }
  // if only far child is intersecting
  else if (addFar)
  {
    // move far child to near
    node.childLeft  = node.childRight;
    // change intersecting flags
    addNear = true;
    addFar  = false;
  }

  BVHNodeIntersectionData bvhNodeIntersectionData;
  bvhNodeIntersectionData.node    = node;
  bvhNodeIntersectionData.addNear = addNear;
  bvhNodeIntersectionData.addFar  = addFar;

  return bvhNodeIntersectionData;
}

inline void initNodeData(Thread uint* lastNodeIndex, Thread uint *traversalStack, const Thread short* stackBase)
{
#ifdef BVH_STACK_TRAVERSAL_CACHE_LAST_NODE
    (*lastNodeIndex)            = setBVHInternalNodeMarker(false, 0);
#else
    traversalStack[*stackBase]  = setBVHInternalNodeMarker(false, 0);
#endif
}

inline void initNodeDataShared(Shared uint* sharedLeafNodeIndex, Thread short* sharedTop)
{
  sharedLeafNodeIndex[(*sharedTop)++] = setBVHInternalNodeMarker(false, 0);
}

inline void pushNodeData(
  const BVHNodeIntersectionData bvhNodeIntersectionData,
  Thread uint*  lastNodeIndex,
  Thread uint*  traversalStack,
  Thread short* stackTop)
{
#ifdef BVH_STACK_TRAVERSAL_CACHE_LAST_NODE
  if (bvhNodeIntersectionData.addNear)
  {
    if (bvhNodeIntersectionData.addFar)
    {
      traversalStack[(*stackTop)++] = bvhNodeIntersectionData.node.childRight;
    }
    (*lastNodeIndex) = bvhNodeIntersectionData.node.childLeft;
    (*stackTop)++;
  }
#else
  if (bvhNodeIntersectionData.addFar)   traversalStack[(*stackTop)++] = bvhNodeIntersectionData.node.childRight;
  if (bvhNodeIntersectionData.addNear)  traversalStack[(*stackTop)++] = bvhNodeIntersectionData.node.childLeft;
#endif
}

inline void pushNodeDataShared(
  BVHNodeIntersectionData bvhNodeIntersectionData,
  Thread uint*  lastNodeIndex,
  Thread uint*  traversalStack,
  Thread short* stackTop,
  Shared uint*  sharedLeafNodeIndex,
  Thread short* sharedTop,
  bool          diverged)
{
  const bool isFirst = simdIsFirst();

  if (bvhNodeIntersectionData.addFar && !diverged)
  {
    const uint child = bvhNodeIntersectionData.node.childRight;
    if (simdAll(simdFirst(child) == child))
    {
      if (isFirst)
      {
        sharedLeafNodeIndex[*sharedTop] = child;
      }
      (*sharedTop)++;
      bvhNodeIntersectionData.addFar = false;
    }
    else
    {
      diverged = true;
    }
  }

  if (bvhNodeIntersectionData.addNear && !diverged)
  {
    const uint child = bvhNodeIntersectionData.node.childLeft;
    if (simdAll(simdFirst(child) == child))
    {
      if (isFirst)
      {
        sharedLeafNodeIndex[*sharedTop] = child;
      }
      (*sharedTop)++;
    }
    else
    {
      diverged = true;
    }
  }

  if (diverged)
  {
    pushNodeData(bvhNodeIntersectionData, lastNodeIndex, traversalStack, stackTop);
  }
}

inline uint popNodeData(
  Thread uint*  lastNodeIndex,
  Thread uint*  traversalStack,
  Thread short* stackTop)
{
#ifdef BVH_STACK_TRAVERSAL_CACHE_LAST_NODE
  --(*stackTop);
  const uint currNodeIndex = ((*lastNodeIndex) != -1) ? (*lastNodeIndex) : traversalStack[*stackTop];
  (*lastNodeIndex) = -1;
  return currNodeIndex;
#else
  return traversalStack[--(*stackTop)];
#endif
}

inline uint popNodeDataShared(
  Thread uint*        lastNodeIndex,
  Thread uint*        traversalStack,
  Thread short*       stackTop,
  const Thread short* stackBase,
  Shared uint*        sharedLeafNodeIndex,
  Thread short*       sharedTop)
{
  if ((*stackTop) > (*stackBase)) return popNodeData(lastNodeIndex, traversalStack, stackTop);
  else                            return sharedLeafNodeIndex[--(*sharedTop)];
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
  Thread uint*                  traversalStack,
  const short                   stackBase,
  Shared uint*                  sharedLeafNodeIndex,
  const short                   sharedBase)
{
  short stackTop = stackBase+1;
  uint lastNodeIndex = -1;

  initNodeData(&lastNodeIndex, traversalStack, &stackBase);

  // break if the stack is empty
  while (stackTop > stackBase)
  {
    const uint currNodeIndex = popNodeData(&lastNodeIndex, traversalStack, &stackTop);

    if (isBVHLeafNode(currNodeIndex))
    {
      if (earliestIntersection(hit, currNodeIndex, rayOrigin, rayDirection, invRayDirection, sign, vertexArray, attributeArray, systemSettings, primInfo))
      {
#ifdef IntersectionTypeAny
        break;
#endif
      }
      continue;
    }

    const BVHNodeIntersectionData bvhNodeIntersectionData = getNodeIntersectionData(hit, rayOrigin, invRayDirection, sign,
      treeInternalNodes, leafParentNodeIndices, nodeParentNodeIndices, treeLeafNodeBoundingBoxes, treeInternalNodeBoundingBoxes, currNodeIndex);

    pushNodeData(bvhNodeIntersectionData, &lastNodeIndex, traversalStack, &stackTop);
  }

  return *hit;
}

inline HitStruct stackTraverseBinaryTreeWithInputsShared(
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
  Thread uint*                  traversalStack,
  const short                   stackBase,
  Shared uint*                  sharedLeafNodeIndex,
  const short                   sharedBase)
{
  short stackTop  = stackBase;
  short sharedTop = sharedBase;

  const uint initialActiveThreads = ((size_t)simd_active_threads_mask()) & 0xFFFFFFFF;
  uint lastNodeIndex = -1;

  initNodeDataShared(sharedLeafNodeIndex, &sharedTop);

  // break if the stack is empty
  while (stackTop > stackBase || sharedTop > sharedBase)
  {
    const uint currNodeIndex = popNodeDataShared(&lastNodeIndex, traversalStack, &stackTop, &stackBase, sharedLeafNodeIndex, &sharedTop);

    if (isBVHLeafNode(currNodeIndex))
    {
      if (earliestIntersection(hit, currNodeIndex, rayOrigin, rayDirection, invRayDirection, sign, vertexArray, attributeArray, systemSettings, primInfo))
      {
#ifdef IntersectionTypeAny
        break;
#endif
      }
      continue;
    }

    const BVHNodeIntersectionData bvhNodeIntersectionData = getNodeIntersectionData(hit, rayOrigin, invRayDirection, sign,
      treeInternalNodes, leafParentNodeIndices, nodeParentNodeIndices, treeLeafNodeBoundingBoxes, treeInternalNodeBoundingBoxes, currNodeIndex);

    const bool diverged = (stackTop > stackBase) || (initialActiveThreads != (((size_t)simd_active_threads_mask()) & 0xFFFFFFFF));
    pushNodeDataShared(bvhNodeIntersectionData, &lastNodeIndex, traversalStack, &stackTop, sharedLeafNodeIndex, &sharedTop, diverged);
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
  uint traversalStack[64];
  HitStruct hit = defaultHit(currentTime);

  return stackTraverseBinaryTreeWithInputs(&hit, treeInternalNodes, leafParentNodeIndices, nodeParentNodeIndices, treeLeafNodeBoundingBoxes, treeInternalNodeBoundingBoxes,
    rayOrigin, rayDirection, invRayDirection, sign, vertexArray, attributeArray, systemSettings, primInfo, traversalStack, 0, sharedLeafNodeIndex, 0);
}

#endif
