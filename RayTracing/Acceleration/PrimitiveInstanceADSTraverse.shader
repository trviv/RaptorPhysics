#ifndef PRIMITIVE_INSTANCE_ACCELERATION_DATA_STRUCT_TRAVERSE_SHADER
#define PRIMITIVE_INSTANCE_ACCELERATION_DATA_STRUCT_TRAVERSE_SHADER

struct BVHTraversalState
{
  uint traverseState;
  uint currNodeIndex;
  uint parentNodeIndex;
  uint nextNodeIndex;
};

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

inline short stackTraverseInstancedBinaryTree(
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
  Thread uint*              traversalStack,
  Thread short              stackTop,
  Thread uint*              primitiveADSInstance)
{
  HitStruct hit = defaultHit(currentTime);

  // break if the stack is empty
  while (stackTop > 0)
  {
    uint currNodeIndex = traversalStack[--stackTop];

    if (isBVHLeafNode(currNodeIndex))
    {
      *primitiveADSInstance = currNodeIndex;
      return stackTop;
    }

    // traverse while a leaf node is found
    const BVHNodeInfo node = treeInternalNodes[removeBVHInternalNodeMarker(currNodeIndex)];

    XAB leftBoundingBox;
    if (isBVHLeafNode(node.childLeft))  leftBoundingBox = treeLeafNodeBoundingBoxes[node.childLeft];
    else                                leftBoundingBox = treeInternalNodeBoundingBoxes[removeBVHInternalNodeMarker(node.childLeft)];

    float leftDist = hit.distance;
    const bool addNear = rayXABIntersectEarliest(&leftDist, leftBoundingBox, rayOrigin, invRayDirection, sign);
    addBVHHit(hit.bvhHits, addNear);

    XAB rightBoundingBox;
    if (isBVHLeafNode(node.childRight)) rightBoundingBox = treeLeafNodeBoundingBoxes[node.childRight];
    else                                rightBoundingBox = treeInternalNodeBoundingBoxes[removeBVHInternalNodeMarker(node.childRight)];

    float rightDist = hit.distance;
    const bool addFar = rayXABIntersectEarliest(&rightDist, rightBoundingBox, rayOrigin, invRayDirection, sign);
    addBVHHit(hit.bvhHits, addFar);

    const bool swapChilds = addNear && addFar && leftDist < rightDist;

    if (addNear) traversalStack[stackTop++] = getChildNode(node, swapChilds);
    if (addFar)  traversalStack[stackTop++] = getChildNode(node, !swapChilds);
  }

  return -1;
}

Kernel void intersectRaysBVHPrimitiveInstancesFlattened(
  Device HitStruct*                       hits,
  const Device RayStruct*                 rays,
  constantKernelInput(uint,               rayCount),
  const Device BVHNodeInfo*               treeInternalNodes,
  const Device uint*                      leafParentNodeIndices,
  const Device uint*                      nodeParentNodeIndices,
  const Device XAB*                       treeLeafNodeBoundingBoxes,
  const Device XAB*                       treeInternalNodeBoundingBoxes,
  const Device PrimitiveADSResources*     primitiveResources,
  const Device float4x4*                  primitiveInstanceTransforms,
  const Device PrimitiveInstanceADSLeaf*  primitiveInstanceNodes,
  constantKernelInput(uint,               primitiveCount),
  atomicKernelInput(uint,                 rayIndexAtomicBuffer),
  sharedMemKernelInput(uint,              sharedLeafNodeIndex, 13)
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
    primitiveResources[0].treeInternalNodes,
    primitiveResources[0].leafParentNodeIndices,
    primitiveResources[0].nodeParentNodeIndices,
    primitiveResources[0].treeLeafNodeBoundingBoxes,
    primitiveResources[0].treeInternalNodeBoundingBoxes,
    ray.origin, ray.direction, invRayDirection, sign,
    primitiveResources[0].vertexArray,
    primitiveResources[0].attributeArray,
    primitiveResources[0].systemSettings,
    &primInfo, threadLocalIndex(), sharedLeafNodeIndex);

  traversalSetHitNormal(ray, &hit, primitiveResources[0].vertexArray, primitiveResources[0].attributeArray, primitiveResources[0].vertexAttributeArray, primitiveResources[0].systemSettings);
  traversalStoreHit(&hits[index], hit);
}

Kernel void intersectRaysBVHPrimitiveInstances(
  Device HitStruct*                       hits,
  const Device RayStruct*                 rays,
  constantKernelInput(uint,               rayCount),
  const Device BVHNodeInfo*               treeInternalNodes,
  const Device uint*                      leafParentNodeIndices,
  const Device uint*                      nodeParentNodeIndices,
  const Device XAB*                       treeLeafNodeBoundingBoxes,
  const Device XAB*                       treeInternalNodeBoundingBoxes,
  const Device PrimitiveADSResources*     primitiveResources,
  const Device float4x4*                  primitiveInstanceTransforms,
  const Device PrimitiveInstanceADSLeaf*  primitiveInstanceNodes,
  constantKernelInput(uint,               primitiveCount),
  atomicKernelInput(uint,                 rayIndexAtomicBuffer),
  sharedMemKernelInput(uint,              sharedLeafNodeIndex, 13)
  KERNEL_THREAD_ARGUMENTS
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint index = threadIndex();

  if (index >= rayCount)
    return;

  HitStruct finalHit = defaultHit(rays[index].maxDistance);

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

  do
  {
    stacklessTraverseInstancedBinaryTree(finalHit.distance,
      treeInternalNodes,
      leafParentNodeIndices,
      nodeParentNodeIndices,
      treeLeafNodeBoundingBoxes,
      treeInternalNodeBoundingBoxes,
      worldRay.origin, worldRay.direction, worldInvRayDirection, worldSign,
      localIndex, sharedLeafNodeIndex, traversalState);

    if (BVH_TRAVERSAL_TRAVERSE_STATE == 0) break;

    const uint primitiveInstance = BVH_TRAVERSAL_CURRENT_NODE;
    const uint primitiveADSIndex = primitiveInstanceNodes[primitiveInstance].primitiveADSIndex;
    const IdentityInfo primitiveIdentity = primitiveInstanceNodes[primitiveInstance].primitiveIdentity;
    const float4x4 invTransform = primitiveInstanceTransforms[primitiveCount + primitiveInstance];
    const PrimitiveADSResources primitiveADSResource = primitiveResources[primitiveADSIndex];

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
      setHitNormal(finalHit.normal, normalize(mulMatrixVec(primitiveInstanceTransforms[primitiveInstance], constructFloat4(finalHit.normal, 0.f)).xyz));

//#ifdef IntersectionTypeAny
//      break;
//#endif
    }
  }
  while (true);

  traversalStoreHit(&hits[index], finalHit);
}

Kernel void intersectRaysBVHPrimitiveInstancesStacked(
  Device HitStruct*                       hits,
  const Device RayStruct*                 rays,
  constantKernelInput(uint,               rayCount),
  const Device BVHNodeInfo*               treeInternalNodes,
  const Device uint*                      leafParentNodeIndices,
  const Device uint*                      nodeParentNodeIndices,
  const Device XAB*                       treeLeafNodeBoundingBoxes,
  const Device XAB*                       treeInternalNodeBoundingBoxes,
  const Device PrimitiveADSResources*     primitiveResources,
  const Device float4x4*                  primitiveInstanceTransforms,
  const Device PrimitiveInstanceADSLeaf*  primitiveInstanceNodes,
  constantKernelInput(uint,               primitiveCount),
  atomicKernelInput(uint,                 rayIndexAtomicBuffer),
  sharedMemKernelInput(uint,              sharedLeafNodeIndex, 13)
  KERNEL_THREAD_ARGUMENTS
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint index = threadIndex();

  if (index >= rayCount)
    return;

  HitStruct finalHit = defaultHit(rays[index].maxDistance);

  const ushort localIndex = threadLocalIndex();

  const RayStruct worldRay = rays[index];
  const float3 worldInvRayDirection = 1.f / worldRay.direction;
  const bool3 worldSign = selectInput3(worldRay.direction < 0.f);

  uint traversalStack[48];
  traversalStack[0] = setBVHInternalNodeMarker(false, 0);

  short stackTop = 1;
  uint primitiveInstance = -1;

  do
  {
    stackTop = stackTraverseInstancedBinaryTree(finalHit.distance,
      treeInternalNodes,
      leafParentNodeIndices,
      nodeParentNodeIndices,
      treeLeafNodeBoundingBoxes,
      treeInternalNodeBoundingBoxes,
      worldRay.origin, worldRay.direction, worldInvRayDirection, worldSign,
      traversalStack, stackTop, &primitiveInstance);

    if (stackTop == -1) break;

    const uint primitiveADSIndex = primitiveInstanceNodes[primitiveInstance].primitiveADSIndex;
    const IdentityInfo primitiveIdentity = primitiveInstanceNodes[primitiveInstance].primitiveIdentity;
    const float4x4 invTransform = primitiveInstanceTransforms[primitiveCount + primitiveInstance];
    const PrimitiveADSResources primitiveADSResource = primitiveResources[primitiveADSIndex];

    RayStruct localRay;
    localRay.origin = mulMatrixVec(invTransform, constructFloat4(worldRay.origin, 1.f)).xyz;
    localRay.direction = mulMatrixVec(invTransform, constructFloat4(worldRay.direction, 0.f)).xyz;

    const float3 localInvRayDirection = 1.f / localRay.direction;
    const bool3 localSign = selectInput3(localRay.direction < 0.f);

    DecodedPrimitiveInfo primInfo = defaultPrimitiveInfo();

    const HitStruct hit = stackTraverseBinaryTreeWithInput(finalHit.distance,
      primitiveADSResource.treeInternalNodes,
      primitiveADSResource.leafParentNodeIndices,
      primitiveADSResource.nodeParentNodeIndices,
      primitiveADSResource.treeLeafNodeBoundingBoxes,
      primitiveADSResource.treeInternalNodeBoundingBoxes,
      localRay.origin, localRay.direction, localInvRayDirection, localSign,
      primitiveADSResource.vertexArray,
      primitiveADSResource.attributeArray,
      primitiveADSResource.systemSettings,
      &primInfo, localIndex, sharedLeafNodeIndex, stackTop, traversalStack);

    if (hit.distance < finalHit.distance)
    {
      finalHit = hit;
      setHitPrimitiveIdentity(finalHit.primitiveIdentity, primitiveIdentity);
      traversalSetHitNormal(localRay, &finalHit, primitiveADSResource.vertexArray, primitiveADSResource.attributeArray, primitiveADSResource.vertexAttributeArray, primitiveADSResource.systemSettings, false);
      setHitNormal(finalHit.normal, normalize(mulMatrixVec(primitiveInstanceTransforms[primitiveInstance], constructFloat4(finalHit.normal, 0.f)).xyz));

#ifdef IntersectionTypeAny
      break;
#endif
    }
  }
  while (true);

  traversalStoreHit(&hits[index], finalHit);
}

Kernel void intersectRaysBVHPrimitiveInstancesSingleTraversal(
  Device HitStruct*                       hits,
  const Device RayStruct*                 rays,
  constantKernelInput(uint,               rayCount),
  const Device BVHNodeInfo*               treeInternalNodes,
  const Device uint*                      leafParentNodeIndices,
  const Device uint*                      nodeParentNodeIndices,
  const Device XAB*                       treeLeafNodeBoundingBoxes,
  const Device XAB*                       treeInternalNodeBoundingBoxes,
  const Device PrimitiveADSResources*     primitiveResources,
  const Device float4x4*                  primitiveInstanceTransforms,
  const Device PrimitiveInstanceADSLeaf*  primitiveInstanceNodes,
  constantKernelInput(uint,               primitiveCount),
  atomicKernelInput(uint,                 rayIndexAtomicBuffer),
  sharedMemKernelInput(uint,              sharedLeafNodeIndex, 13)
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

  short stackTop = 1;
  uint traversalStack[48];

  traversalStack[0] = setBVHInternalNodeMarker(false, 0);

  RayStruct ray = worldRay;
  float3 invRayDirection = worldInvRayDirection;
  bool3 sign = worldSign;

  uint primitiveInstance;
  IdentityInfo primitiveIdentity;
  PrimitiveADSResources primitiveADSResource;

  primitiveADSResource.treeInternalNodes = treeInternalNodes;
  primitiveADSResource.treeLeafNodeBoundingBoxes = treeLeafNodeBoundingBoxes;
  primitiveADSResource.treeInternalNodeBoundingBoxes = treeInternalNodeBoundingBoxes;

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

      primitiveADSResource.treeInternalNodes = treeInternalNodes;
      primitiveADSResource.treeLeafNodeBoundingBoxes = treeLeafNodeBoundingBoxes;
      primitiveADSResource.treeInternalNodeBoundingBoxes = treeInternalNodeBoundingBoxes;

      continue;
    }

    uint currNodeIndex = traversalStack[--stackTop];

    if (isBVHLeafNode(currNodeIndex))
    {
      if (!isTopTree)
      {
        if (earliestIntersection(&hit, currNodeIndex, ray.origin, ray.direction, invRayDirection, sign, primitiveADSResource.vertexArray, primitiveADSResource.attributeArray, primitiveADSResource.systemSettings, &primInfo))
        {
          setHitPrimitiveIdentity(hit.primitiveIdentity, primitiveIdentity);
          traversalSetHitNormal(ray, &hit, primitiveADSResource.vertexArray, primitiveADSResource.attributeArray, primitiveADSResource.vertexAttributeArray, primitiveADSResource.systemSettings, false);
          setHitNormal(hit.normal, normalize(mulMatrixVec(primitiveInstanceTransforms[primitiveInstance], constructFloat4(hit.normal, 0.f)).xyz));

#ifdef IntersectionTypeAny
          break;
#endif
        }
      }
      else
      {
        primitiveInstance = currNodeIndex;
        uint primitiveADSIndex = primitiveInstanceNodes[primitiveInstance].primitiveADSIndex;
        primitiveIdentity = primitiveInstanceNodes[primitiveInstance].primitiveIdentity;
        float4x4 invTransform = primitiveInstanceTransforms[primitiveCount + primitiveInstance];
        primitiveADSResource = primitiveResources[primitiveADSIndex];

        ray.origin = mulMatrixVec(invTransform, constructFloat4(worldRay.origin, 1.f)).xyz;
        ray.direction = mulMatrixVec(invTransform, constructFloat4(worldRay.direction, 0.f)).xyz;

        invRayDirection = 1.f / ray.direction;
        sign = selectInput3(ray.direction < 0.f);

        isTopTree = false;
        stackOffset = stackTop;
        traversalStack[stackTop++] = setBVHInternalNodeMarker(false, 0);

        primInfo = defaultPrimitiveInfo();
      }
      continue;
    }

    // traverse while a leaf node is found
    const BVHNodeInfo node = primitiveADSResource.treeInternalNodes[removeBVHInternalNodeMarker(currNodeIndex)];

    XAB leftBoundingBox;
    if (isBVHLeafNode(node.childLeft))  leftBoundingBox = primitiveADSResource.treeLeafNodeBoundingBoxes[node.childLeft];
    else                                leftBoundingBox = primitiveADSResource.treeInternalNodeBoundingBoxes[removeBVHInternalNodeMarker(node.childLeft)];

    float leftDist = hit.distance;
    const bool addNear = rayXABIntersectEarliest(&leftDist, leftBoundingBox, ray.origin, invRayDirection, sign);
    addBVHHit(hit.bvhHits, addNear);

    XAB rightBoundingBox;
    if (isBVHLeafNode(node.childRight)) rightBoundingBox = primitiveADSResource.treeLeafNodeBoundingBoxes[node.childRight];
    else                                rightBoundingBox = primitiveADSResource.treeInternalNodeBoundingBoxes[removeBVHInternalNodeMarker(node.childRight)];

    float rightDist = hit.distance;
    const bool addFar = rayXABIntersectEarliest(&rightDist, rightBoundingBox, ray.origin, invRayDirection, sign);
    addBVHHit(hit.bvhHits, addFar);

    const bool swapChilds = addNear && addFar && leftDist < rightDist;

    if (addNear) traversalStack[stackTop++] = getChildNode(node, swapChilds);
    if (addFar)  traversalStack[stackTop++] = getChildNode(node, !swapChilds);
  }

  traversalStoreHit(&hits[index], hit);
}

#endif
