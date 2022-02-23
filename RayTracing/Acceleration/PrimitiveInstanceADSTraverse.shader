#ifndef PRIMITIVE_INSTANCE_ACCELERATION_DATA_STRUCT_TRAVERSE_SHADER
#define PRIMITIVE_INSTANCE_ACCELERATION_DATA_STRUCT_TRAVERSE_SHADER

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

  HitStruct hit = stacklessTraverseBinaryTree(rays[index].maxDistance,
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
    stacklessTraverseBinaryTree(finalHit.distance,
      treeInternalNodes,
      leafParentNodeIndices,
      nodeParentNodeIndices,
      treeLeafNodeBoundingBoxes,
      treeInternalNodeBoundingBoxes,
      worldRay.origin, worldRay.direction, worldInvRayDirection, worldSign,
      0, 0, 0, 0, localIndex, sharedLeafNodeIndex, true, traversalState);

    if (BVH_TRAVERSAL_TRAVERSE_STATE == 0) break;

    const uint primitiveInstance = BVH_TRAVERSAL_CURRENT_NODE;
    const uint primitiveADSIndex = primitiveInstanceNodes[primitiveInstance].primitiveADSIndex;
    const float4x4 invTransform = primitiveInstanceTransforms[primitiveCount + primitiveInstance];
    const PrimitiveADSResources primitiveADSResource = primitiveResources[primitiveADSIndex];

    RayStruct localRay;
    localRay.origin = mulMatrixVec(invTransform, constructFloat4(worldRay.origin, 1.f)).xyz;
    localRay.direction = mulMatrixVec(invTransform, constructFloat4(worldRay.direction, 0.f)).xyz;

    const float3 localInvRayDirection = 1.f / localRay.direction;
    const bool3 localSign = selectInput3(localRay.direction < 0.f);

    DecodedPrimitiveInfo primInfo = defaultPrimitiveInfo();

    const HitStruct hit = stacklessTraverseBinaryTree(finalHit.distance,
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
      setHitPrimitiveIdentity(finalHit.primitiveIdentity, primitiveInstanceNodes[primitiveInstance].primitiveIdentity);
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

#endif
