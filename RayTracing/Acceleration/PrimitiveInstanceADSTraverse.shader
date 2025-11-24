/*
 * RaptorPhysics
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#ifndef PRIMITIVE_INSTANCE_ACCELERATION_DATA_STRUCT_TRAVERSE_SHADER
#define PRIMITIVE_INSTANCE_ACCELERATION_DATA_STRUCT_TRAVERSE_SHADER

#include "BoundingVolumeHierarchyADSTraverse.shader"

typedef struct
{
  uint traverseState;
  uint currNodeIndex;
  uint parentNodeIndex;
  uint nextNodeIndex;
} BVHTraversalState;

#if RAY_TRAVERSAL_BVH_MAX_LEAFS == 0
#undef TRAVERSAL_STATE_IN_SHARED_MEMORY
#endif

#ifdef TRAVERSAL_STATE_IN_SHARED_MEMORY
#define BVH_TRAVERSAL_TRAVERSE_STATE  leafNodeIndex[localIndex * RAY_TRAVERSAL_SHARED_MEMORY_INDEX_STRIDE + RAY_TRAVERSAL_BVH_MAX_LEAFS]
#define BVH_TRAVERSAL_NEXT_NODE       leafNodeIndex[localIndex * RAY_TRAVERSAL_SHARED_MEMORY_INDEX_STRIDE + RAY_TRAVERSAL_BVH_MAX_LEAFS + 1]
#define BVH_TRAVERSAL_PARENT_NODE     leafNodeIndex[localIndex * RAY_TRAVERSAL_SHARED_MEMORY_INDEX_STRIDE + RAY_TRAVERSAL_BVH_MAX_LEAFS + 2]
#define BVH_TRAVERSAL_CURRENT_NODE    leafNodeIndex[localIndex * RAY_TRAVERSAL_SHARED_MEMORY_INDEX_STRIDE + RAY_TRAVERSAL_BVH_MAX_LEAFS + 3]
#else
#define BVH_TRAVERSAL_TRAVERSE_STATE  traversalState->traverseState
#define BVH_TRAVERSAL_NEXT_NODE       traversalState->nextNodeIndex
#define BVH_TRAVERSAL_PARENT_NODE     traversalState->parentNodeIndex
#define BVH_TRAVERSAL_CURRENT_NODE    traversalState->currNodeIndex
#endif

inline HitStruct stacklessTraverseInstancedBinaryTree(
  float                     currentTime,
  const Device BVHNodeInfo* treeInternalNodes,
  const Device uint*        leafParentNodeIndices,
  const Device uint*        nodeParentNodeIndices,
  const Device XAB*         treeLeafNodeBoundingBoxes,
  const Device XAB*         treeInternalNodeBoundingBoxes,
  const float3              rayOrigin,
  const float3              rayDirection,
  const float3              invRayDirection,
  const bool3               sign,
  const ushort              localIndex,
  Shared uint*              sharedLeafNodeIndex,
  Thread BVHTraversalState* traversalState)
{
  HitStruct hit          = defaultHit(currentTime);
  const uint rootNode    = setBVHInternalNodeMarker(false, 0);

  ushort traverseState;
  BVHNodeInfo parentNode;

  uint parentNodeIndex;
  uint currNodeIndex;
  ushort nearPlane;

#if RAY_TRAVERSAL_SHARED_MEMORY_INDEX_STRIDE > 0
  Shared uint *leafNodeIndex = sharedLeafNodeIndex;
#elif RAY_TRAVERSAL_BVH_MAX_LEAFS > 0
  uint leafNodeIndex[RAY_TRAVERSAL_BVH_MAX_LEAFS];
#endif

  if (BVH_TRAVERSAL_TRAVERSE_STATE != 0)
  {
    traverseState   = BVH_TRAVERSAL_TRAVERSE_STATE;
    currNodeIndex   = BVH_TRAVERSAL_NEXT_NODE;
    parentNodeIndex = BVH_TRAVERSAL_PARENT_NODE;
    if (parentNodeIndex != rootNode)
    {
      parentNode    = treeInternalNodes[removeBVHInternalNodeMarker(parentNodeIndex)];
    }
    nearPlane       = getNearChilds(treeLeafNodeBoundingBoxes, treeInternalNodeBoundingBoxes, parentNode, rayOrigin, invRayDirection, currentTime, sign).z;
    BVH_TRAVERSAL_TRAVERSE_STATE = 0;
  }
  else
  {
    traverseState   = BVH_TRAVERSAL_FROM_PARENT;
    parentNode      = treeInternalNodes[0];
    parentNodeIndex = rootNode;
    nearPlane       = getNearChilds(treeLeafNodeBoundingBoxes, treeInternalNodeBoundingBoxes, parentNode, rayOrigin, invRayDirection, currentTime, sign).z;
    currNodeIndex   = parentNode.child[nearPlane];
  }

  // main intersection loop
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

    if (intersectsBVH)
    {
      // if leaf mark for test
      if (isLeaf)
      {
        BVH_TRAVERSAL_TRAVERSE_STATE  = traverseState;
        BVH_TRAVERSAL_NEXT_NODE       = nextNodeIndex;
        BVH_TRAVERSAL_PARENT_NODE     = parentNodeIndex;
        BVH_TRAVERSAL_CURRENT_NODE    = currNodeIndex;
        return hit;
      }
      // if internal node move to the node
      else
      {
        // if internal node test bounding box for intersection
        addBVHHit(hit.bvhHits, 1);

        parentNode      = treeInternalNodes[noNodeCurrNodeIndex];
        nearPlane       = getNearChilds(treeLeafNodeBoundingBoxes, treeInternalNodeBoundingBoxes, parentNode, rayOrigin, invRayDirection, currentTime, sign).z;
        parentNodeIndex = currNodeIndex;
        currNodeIndex   = parentNode.child[nearPlane];
        traverseState   = BVH_TRAVERSAL_FROM_PARENT;

        continue;
      }
    }
    currNodeIndex = nextNodeIndex;
  }

  return hit;
}

inline uint stackTraverseInstancedBinaryTree(
  float                     currentTime,
  const Device BVHNodeInfo* treeInternalNodes,
  const Device uint*        leafParentNodeIndices,
  const Device uint*        nodeParentNodeIndices,
  const Device XAB*         treeLeafNodeBoundingBoxes,
  const Device XAB*         treeInternalNodeBoundingBoxes,
  const float3              rayOrigin,
  const float3              rayDirection,
  const float3              invRayDirection,
  const bool3               sign,
  const ushort              localIndex,
  Thread uint*              traversalStack,
  Thread short*             stackTop,
  Shared uint*              sharedLeafNodeIndex,
  Thread short*             sharedTop)
{
  HitStruct hit = defaultHit(currentTime);
  uint lastNodeIndex = -1;

  while ((*stackTop) > 0)
  {
    const uint currNodeIndex = popNodeData(&lastNodeIndex, traversalStack, stackTop);

    if (isBVHLeafNode(currNodeIndex))
    {
      return currNodeIndex;
    }

    const BVHNodeIntersectionData bvhNodeIntersectionData = getNodeIntersectionData(&hit, rayOrigin, invRayDirection, sign,
      treeInternalNodes, leafParentNodeIndices, nodeParentNodeIndices, treeLeafNodeBoundingBoxes, treeInternalNodeBoundingBoxes, currNodeIndex);

    pushNodeData(bvhNodeIntersectionData, &lastNodeIndex, traversalStack, stackTop);
  }

  return -1;
}

inline uint stackTraverseInstancedBinaryTreeShared(
  float                     currentTime,
  const Device BVHNodeInfo* treeInternalNodes,
  const Device uint*        leafParentNodeIndices,
  const Device uint*        nodeParentNodeIndices,
  const Device XAB*         treeLeafNodeBoundingBoxes,
  const Device XAB*         treeInternalNodeBoundingBoxes,
  const float3              rayOrigin,
  const float3              rayDirection,
  const float3              invRayDirection,
  const bool3               sign,
  const ushort              localIndex,
  Thread uint*              traversalStack,
  Thread short*             stackTop,
  Shared uint*              sharedLeafNodeIndex,
  Thread short*             sharedTop)
{
  HitStruct hit = defaultHit(currentTime);
  const short stackBase = 0;
  const short sharedBase = (localIndex >> 5) * BVH_TRAVERSAL_SHARED_ELEMENTS;
  const uint initialActiveThreads = ((size_t)simd_active_threads_mask()) & 0xFFFFFFFF;

  uint lastNodeIndex = -1;

  while ((*stackTop) > stackBase || (*sharedTop) > sharedBase)
  {
    const uint currNodeIndex = popNodeDataShared(&lastNodeIndex, traversalStack, stackTop, &stackBase, sharedLeafNodeIndex, sharedTop);

    if (isBVHLeafNode(currNodeIndex))
    {
      return currNodeIndex;
    }

    const BVHNodeIntersectionData bvhNodeIntersectionData = getNodeIntersectionData(&hit, rayOrigin, invRayDirection, sign,
      treeInternalNodes, leafParentNodeIndices, nodeParentNodeIndices, treeLeafNodeBoundingBoxes, treeInternalNodeBoundingBoxes, currNodeIndex);

    const bool diverged = ((*stackTop) > stackBase) || (initialActiveThreads != (((size_t)simd_active_threads_mask()) & 0xFFFFFFFF));
    pushNodeDataShared(bvhNodeIntersectionData, &lastNodeIndex, traversalStack, stackTop, sharedLeafNodeIndex, sharedTop, diverged);
  }

  return -1;
}

Kernel void intersectRaysBVHPrimitiveInstancesFlattened(
  Device HitStruct*                     hits,
  const Device RayStruct*               rays,
  constantKernelInput(uint,             rayCount),
  Const PrimitiveInstanceADSResources*  primitiveInstanceADSResources,
  atomicKernelInput(uint,               rayIndexAtomicBuffer),
  sharedMemKernelInput(uint,            sharedLeafNodeIndex, 5)
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

  HitStruct hit = PRIMITIVE_INSTANCE_ADS_PRIMITIVE_TRAVERSAL(rays[index].maxDistance,
    primitiveInstanceADSResources->primitiveADSResources[0].treeInternalNodes,
    primitiveInstanceADSResources->primitiveADSResources[0].leafParentNodeIndices,
    primitiveInstanceADSResources->primitiveADSResources[0].nodeParentNodeIndices,
    primitiveInstanceADSResources->primitiveADSResources[0].treeLeafNodeBoundingBoxes,
    primitiveInstanceADSResources->primitiveADSResources[0].treeInternalNodeBoundingBoxes,
    ray.origin, ray.direction, invRayDirection, sign,
    primitiveInstanceADSResources->primitiveADSResources[0].vertexArray,
    primitiveInstanceADSResources->primitiveADSResources[0].attributeArray,
    primitiveInstanceADSResources->primitiveADSResources[0].systemSettings,
    &primInfo, threadLocalIndex(), sharedLeafNodeIndex);

  traversalSetHitNormal(ray, &hit,
    primitiveInstanceADSResources->primitiveADSResources[0].vertexArray,
    primitiveInstanceADSResources->primitiveADSResources[0].attributeArray,
    primitiveInstanceADSResources->primitiveADSResources[0].vertexAttributeArray,
    primitiveInstanceADSResources->primitiveADSResources[0].systemSettings);
  traversalStoreHit(&hits[index], hit);
}

Kernel void intersectRaysBVHPrimitiveInstancesStackless(
  Device HitStruct*                     hits,
  const Device RayStruct*               rays,
  constantKernelInput(uint,             rayCount),
  Const PrimitiveInstanceADSResources*  primitiveInstanceADSResources,
  atomicKernelInput(uint,               rayIndexAtomicBuffer),
  sharedMemKernelInput(uint,            sharedLeafNodeIndex, 5)
  KERNEL_THREAD_ARGUMENTS
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint index = threadIndex();

  if (index >= rayCount)
    return;

#if RAY_TRAVERSAL_SHARED_MEMORY_INDEX_STRIDE > 0 && defined(TRAVERSAL_STATE_IN_SHARED_MEMORY)
  Shared uint *leafNodeIndex = sharedLeafNodeIndex;
#endif

  const ushort localIndex = threadLocalIndex();

  Thread BVHTraversalState *traversalState = 0;
#ifndef TRAVERSAL_STATE_IN_SHARED_MEMORY
  BVHTraversalState traversalStateObj;
  traversalState = &traversalStateObj;
#endif

  BVH_TRAVERSAL_TRAVERSE_STATE = 0;

  const RayStruct worldRay = rays[index];
  const float3 worldInvRayDirection = 1.f / worldRay.direction;
  const bool3 worldSign = selectInput3(worldRay.direction < 0.f);

  HitStruct finalHit = defaultHit(worldRay.maxDistance);

  do
  {
    stacklessTraverseInstancedBinaryTree(finalHit.distance,
      primitiveInstanceADSResources->treeInternalNodes,
      primitiveInstanceADSResources->leafParentNodeIndices,
      primitiveInstanceADSResources->nodeParentNodeIndices,
      primitiveInstanceADSResources->treeLeafNodeBoundingBoxes,
      primitiveInstanceADSResources->treeInternalNodeBoundingBoxes,
      worldRay.origin, worldRay.direction, worldInvRayDirection, worldSign,
      localIndex, sharedLeafNodeIndex, traversalState);

    if (BVH_TRAVERSAL_TRAVERSE_STATE == 0) break;

    const uint primitiveInstance = BVH_TRAVERSAL_CURRENT_NODE;
    const uint primitiveADSIndex = primitiveInstanceADSResources->primitiveInstanceNodes[primitiveInstance].primitiveADSIndex;
    const IdentityInfo primitiveIdentity = primitiveInstanceADSResources->primitiveInstanceNodes[primitiveInstance].primitiveIdentity;
    const float4x4 invTransform = primitiveInstanceADSResources->primitiveInstanceTransforms[primitiveInstanceADSResources->primitiveCount + primitiveInstance];
    const PrimitiveADSResources primitiveADSResource = primitiveInstanceADSResources->primitiveADSResources[primitiveADSIndex];

    RayStruct localRay;
    localRay.origin = mulMatrixVec(invTransform, constructFloat4(worldRay.origin, 1.f)).xyz;
    localRay.direction = mulMatrixVec(invTransform, constructFloat4(worldRay.direction, 0.f)).xyz;

    const float3 localInvRayDirection = 1.f / localRay.direction;
    const bool3 localSign = selectInput3(localRay.direction < 0.f);

    DecodedPrimitiveInfo primInfo = defaultPrimitiveInfo();

    const HitStruct hit = PRIMITIVE_INSTANCE_ADS_PRIMITIVE_TRAVERSAL(finalHit.distance,
      primitiveADSResource.treeInternalNodes,
      primitiveADSResource.leafParentNodeIndices,
      primitiveADSResource.nodeParentNodeIndices,
      primitiveADSResource.treeLeafNodeBoundingBoxes,
      primitiveADSResource.treeInternalNodeBoundingBoxes,
      localRay.origin, localRay.direction, localInvRayDirection, localSign,
      primitiveADSResource.vertexArray,
      primitiveADSResource.attributeArray,
      primitiveADSResource.systemSettings,
      &primInfo, localIndex, sharedLeafNodeIndex);

    if (hit.distance < finalHit.distance)
    {
      finalHit = hit;
      setHitPrimitiveIdentity(finalHit.primitiveIdentity, primitiveIdentity);
      traversalSetHitNormal(localRay, &finalHit, primitiveADSResource.vertexArray, primitiveADSResource.attributeArray, primitiveADSResource.vertexAttributeArray, primitiveADSResource.systemSettings, false);
      setHitNormal(finalHit.normal, normalize(mulMatrixVec(primitiveInstanceADSResources->primitiveInstanceTransforms[primitiveInstance], constructFloat4(finalHit.normal, 0.f)).xyz));

#ifdef IntersectionTypeAny
      break;
#endif
    }
  }
  while (true);

  traversalStoreHit(&hits[index], finalHit);
}

inline HitStruct intersectRayBVHPrimitiveInstancesStacked(
  const RayStruct                       worldRay,
  Const PrimitiveInstanceADSResources*  primitiveInstanceADSResources,
  Shared uint*                          sharedTraversalNodes = 0,
  const short                           localIndex           = 0)
{
  const float3 worldInvRayDirection = 1.f / worldRay.direction;
  const bool3 worldSign = selectInput3(worldRay.direction < 0.f);

  HitStruct finalHit = defaultHit(worldRay.maxDistance);

  short stackTop = 0;
  uint traversalStack[64];

  float minTime = finalHit.distance;

  short sharedTop = (localIndex >> 5) * BVH_TRAVERSAL_SHARED_ELEMENTS;

#ifdef TRAVERSAL_USES_SHARED_MEMORY
  sharedTraversalNodes[sharedTop++] = setBVHInternalNodeMarker(false, 0);
#else
  traversalStack[stackTop++] = setBVHInternalNodeMarker(false, 0);
#endif

  do
  {
#ifdef TRAVERSAL_USES_SHARED_MEMORY
    const uint primitiveInstance = stackTraverseInstancedBinaryTreeShared(
#else
    const uint primitiveInstance = stackTraverseInstancedBinaryTree(
#endif
      finalHit.distance,
      primitiveInstanceADSResources->treeInternalNodes,
      primitiveInstanceADSResources->leafParentNodeIndices,
      primitiveInstanceADSResources->nodeParentNodeIndices,
      primitiveInstanceADSResources->treeLeafNodeBoundingBoxes,
      primitiveInstanceADSResources->treeInternalNodeBoundingBoxes,
      worldRay.origin, worldRay.direction, worldInvRayDirection, worldSign,
      localIndex, traversalStack, &stackTop, sharedTraversalNodes, &sharedTop);

    if (primitiveInstance == -1) break;

    const uint primitiveADSIndex                     = primitiveInstanceADSResources->primitiveInstanceNodes[primitiveInstance].primitiveADSIndex;
    const IdentityInfo primitiveIdentity             = primitiveInstanceADSResources->primitiveInstanceNodes[primitiveInstance].primitiveIdentity;
    const float4x4 invTransform                      = primitiveInstanceADSResources->primitiveInstanceTransforms[primitiveInstanceADSResources->primitiveCount + primitiveInstance];
    const PrimitiveADSResources primitiveADSResource = primitiveInstanceADSResources->primitiveADSResources[primitiveADSIndex];

    RayStruct localRay;
    localRay.origin = mulMatrixVec(invTransform, constructFloat4(worldRay.origin, 1.f)).xyz;
    localRay.direction = mulMatrixVec(invTransform, constructFloat4(worldRay.direction, 0.f)).xyz;

    const float3 localInvRayDirection = 1.f / localRay.direction;
    const bool3 localSign = selectInput3(localRay.direction < 0.f);

    DecodedPrimitiveInfo primInfo = defaultPrimitiveInfo();

#ifdef TRAVERSAL_USES_SHARED_MEMORY
    stackTraverseBinaryTreeWithInputsShared(&finalHit,
#else
    stackTraverseBinaryTreeWithInputs(&finalHit,
#endif
      primitiveADSResource.treeInternalNodes,
      primitiveADSResource.leafParentNodeIndices,
      primitiveADSResource.nodeParentNodeIndices,
      primitiveADSResource.treeLeafNodeBoundingBoxes,
      primitiveADSResource.treeInternalNodeBoundingBoxes,
      localRay.origin, localRay.direction, localInvRayDirection, localSign,
      primitiveADSResource.vertexArray,
      primitiveADSResource.attributeArray,
      primitiveADSResource.systemSettings,
      &primInfo, traversalStack, stackTop, sharedTraversalNodes, sharedTop);

    if (minTime > finalHit.distance)
    {
      minTime = finalHit.distance;
      setHitPrimitiveIdentity(finalHit.primitiveIdentity, primitiveIdentity);
      traversalSetHitNormal(localRay, &finalHit,
        primitiveADSResource.vertexArray,
        primitiveADSResource.attributeArray,
        primitiveADSResource.vertexAttributeArray,
        primitiveADSResource.systemSettings, false);
      setHitNormal(finalHit.normal, normalize(mulMatrixVec(primitiveInstanceADSResources->primitiveInstanceTransforms[primitiveInstance], constructFloat4(finalHit.normal, 0.f)).xyz));

#ifdef IntersectionTypeAny
      break;
#endif
    }
  }
  while (true);

  return finalHit;
}

Kernel void intersectRaysBVHPrimitiveInstancesSingleTraversal(
  Device HitStruct*                     hits,
  const Device RayStruct*               rays,
  constantKernelInput(uint,             rayCount),
  Const PrimitiveInstanceADSResources*  primitiveInstanceADSResources,
  atomicKernelInput(uint,               rayIndexAtomicBuffer),
  sharedMemKernelInput(uint,            sharedLeafNodeIndex, 5)
  KERNEL_THREAD_ARGUMENTS
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint index = threadIndex();

  if (index >= rayCount)
    return;

  const RayStruct worldRay = rays[index];
  const float3 worldInvRayDirection = 1.f / worldRay.direction;
  const bool3 worldSign = selectInput3(worldRay.direction < 0.f);

  DecodedPrimitiveInfo primInfo;

  bool isTopTree = true;
  short stackOffset = 0;

  HitStruct hit = defaultHit(rays[index].maxDistance);

  const short stackBase = 0;
  short stackTop = stackBase+1;
  uint traversalStack[48];

  uint lastNodeIndex = -1;
  initNodeData(&lastNodeIndex, traversalStack, &stackBase);

  RayStruct ray = worldRay;
  float3 invRayDirection = worldInvRayDirection;
  bool3 sign = worldSign;

  uint primitiveInstance;
  IdentityInfo primitiveIdentity;
  PrimitiveADSResources primitiveADSResource;

  primitiveADSResource.treeInternalNodes              = primitiveInstanceADSResources->treeInternalNodes;
  primitiveADSResource.treeLeafNodeBoundingBoxes      = primitiveInstanceADSResources->treeLeafNodeBoundingBoxes;
  primitiveADSResource.treeInternalNodeBoundingBoxes  = primitiveInstanceADSResources->treeInternalNodeBoundingBoxes;

  // break if the stack is empty
  while (stackTop > stackOffset || !isTopTree)
  {
    if (stackTop == stackOffset && !isTopTree)
    {
      isTopTree = true;
      ray = worldRay;
      invRayDirection = worldInvRayDirection;
      sign = worldSign;
      stackOffset = 0;

      primitiveADSResource.treeInternalNodes              = primitiveInstanceADSResources->treeInternalNodes;
      primitiveADSResource.treeLeafNodeBoundingBoxes      = primitiveInstanceADSResources->treeLeafNodeBoundingBoxes;
      primitiveADSResource.treeInternalNodeBoundingBoxes  = primitiveInstanceADSResources->treeInternalNodeBoundingBoxes;

      continue;
    }

    const uint currNodeIndex = popNodeData(&lastNodeIndex, traversalStack, &stackTop);

    if (isBVHLeafNode(currNodeIndex))
    {
      if (!isTopTree)
      {
        if (earliestIntersection(&hit, currNodeIndex, ray.origin, ray.direction, invRayDirection, sign, primitiveADSResource.vertexArray, primitiveADSResource.attributeArray, primitiveADSResource.systemSettings, &primInfo))
        {
          setHitPrimitiveIdentity(hit.primitiveIdentity, primitiveIdentity);
          traversalSetHitNormal(ray, &hit, primitiveADSResource.vertexArray, primitiveADSResource.attributeArray, primitiveADSResource.vertexAttributeArray, primitiveADSResource.systemSettings, false);
          setHitNormal(hit.normal, normalize(mulMatrixVec(primitiveInstanceADSResources->primitiveInstanceTransforms[primitiveInstance], constructFloat4(hit.normal, 0.f)).xyz));

#ifdef IntersectionTypeAny
          break;
#endif
        }
      }
      else
      {
        primitiveInstance = currNodeIndex;
        uint primitiveADSIndex = primitiveInstanceADSResources->primitiveInstanceNodes[primitiveInstance].primitiveADSIndex;
        primitiveIdentity = primitiveInstanceADSResources->primitiveInstanceNodes[primitiveInstance].primitiveIdentity;
        float4x4 invTransform = primitiveInstanceADSResources->primitiveInstanceTransforms[primitiveInstanceADSResources->primitiveCount + primitiveInstance];
        primitiveADSResource = primitiveInstanceADSResources->primitiveADSResources[primitiveADSIndex];

        ray.origin = mulMatrixVec(invTransform, constructFloat4(worldRay.origin, 1.f)).xyz;
        ray.direction = mulMatrixVec(invTransform, constructFloat4(worldRay.direction, 0.f)).xyz;

        invRayDirection = 1.f / ray.direction;
        sign = selectInput3(ray.direction < 0.f);

        isTopTree = false;
        stackOffset = stackTop;
#ifdef BVH_STACK_TRAVERSAL_CACHE_LAST_NODE
        lastNodeIndex = setBVHInternalNodeMarker(false, 0);
        stackTop++;
#else
        traversalStack[stackTop++] = setBVHInternalNodeMarker(false, 0);
#endif

        primInfo = defaultPrimitiveInfo();
      }
      continue;
    }

    const BVHNodeIntersectionData bvhNodeIntersectionData = getNodeIntersectionData(&hit, ray.origin, invRayDirection, sign,
      primitiveInstanceADSResources->treeInternalNodes,
      primitiveInstanceADSResources->leafParentNodeIndices,
      primitiveInstanceADSResources->nodeParentNodeIndices,
      primitiveInstanceADSResources->treeLeafNodeBoundingBoxes,
      primitiveInstanceADSResources->treeInternalNodeBoundingBoxes, currNodeIndex);

    pushNodeData(bvhNodeIntersectionData, &lastNodeIndex, traversalStack, &stackTop);
  }

  traversalStoreHit(&hits[index], hit);
}

Kernel void intersectRaysPrimitiveInstances(
  Device HitStruct*                     hits,
  const Device RayStruct*               rays,
  constantKernelInput(uint,             rayCount),
  Const PrimitiveInstanceADSResources*  primitiveInstanceADSResources,
  atomicKernelInput(uint,               rayIndexAtomicBuffer),
  sharedMemKernelInput(uint,            sharedLeafNodeIndex, 5)
  KERNEL_THREAD_ARGUMENTS
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint index = threadIndex();

  if (index >= rayCount) return;

  const HitStruct hit = intersectRayBVHPrimitiveInstancesStacked(rays[index], primitiveInstanceADSResources, sharedLeafNodeIndex, threadLocalIndex());
  traversalStoreHit(&hits[index], hit);
}

inline HitStruct intersectRayPrimitiveInstanceADS(
  const RayStruct                       worldRay,
  Const PrimitiveInstanceADSResources*  primitiveInstanceADSResources,
  Shared uint*                          sharedTraversalNodes = 0,
  const short                           localIndex = 0)
{
  return intersectRayBVHPrimitiveInstancesStacked(worldRay, primitiveInstanceADSResources, sharedTraversalNodes, localIndex);
}

#endif
