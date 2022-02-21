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
  const bool3 sign = selectInput3(invRayDirection < 0.f);

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
#ifdef IntersectionTypeClosest
  hits[index] = hit;
#endif
#ifdef IntersectionTypeAny
  setHitPrimitiveIndex(hits[index].primitiveIndex, hit.primitiveIndex);
  setHitPrimitiveIdentity(hits[index].primitiveIdentity, hit.primitiveIdentity);
#endif
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

  HitStruct finalHit;
  initializeHit(&finalHit);

  finalHit.distance = rays[index].maxDistance;

#if RAY_TRAVERSAL_SHARED_MEMORY_INDEX_STRIDE > 0
  Shared uint *leafNodeIndex = sharedLeafNodeIndex;
#else
  uint leafNodeIndex[RAY_TRAVERSAL_BVH_MAX_LEAFS];
#endif

  const ushort localIndex = threadLocalIndex();

  leafNodeIndex[localIndex * RAY_TRAVERSAL_SHARED_MEMORY_INDEX_STRIDE + RAY_TRAVERSAL_BVH_MAX_LEAFS] = -1;

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
      0, 0, 0, 0, localIndex, sharedLeafNodeIndex);

    if (leafNodeIndex[localIndex * RAY_TRAVERSAL_SHARED_MEMORY_INDEX_STRIDE + BVH_TRAVERSAL_TRAVERSE_STATE] == -1) break;

    const uint primitiveADSIndex  = leafNodeIndex[localIndex * RAY_TRAVERSAL_SHARED_MEMORY_INDEX_STRIDE + BVH_TRAVERSAL_CURRENT_NODE];
    const float4x4 invTransform   = primitiveInstanceTransforms[primitiveCount + primitiveADSIndex];

    RayStruct localRay;
    localRay.origin = mulMatrixVec(invTransform, constructFloat4(worldRay.origin, 1.f)).xyz;
    localRay.direction = mulMatrixVec(invTransform, constructFloat4(worldRay.direction, 0.f)).xyz;

    const float3 localInvRayDirection = 1.f / localRay.direction;
    const bool3 localSign = selectInput3(localRay.direction < 0.f);

    DecodedPrimitiveInfo primInfo = defaultPrimitiveInfo();

    const HitStruct hit = stacklessTraverseBinaryTree(finalHit.distance,
      primitiveResources[primitiveADSIndex].treeInternalNodes,
      primitiveResources[primitiveADSIndex].leafParentNodeIndices,
      primitiveResources[primitiveADSIndex].nodeParentNodeIndices,
      primitiveResources[primitiveADSIndex].treeLeafNodeBoundingBoxes,
      primitiveResources[primitiveADSIndex].treeInternalNodeBoundingBoxes,
      localRay.origin, localRay.direction, localInvRayDirection, localSign,
      primitiveResources[primitiveADSIndex].vertexArray,
      primitiveResources[primitiveADSIndex].attributeArray,
      primitiveResources[primitiveADSIndex].systemSettings,
      &primInfo, localIndex, sharedLeafNodeIndex,
      true);

    if (hit.distance < finalHit.distance)
    {
      finalHit = hit;

      setHitPrimitiveIdentity(finalHit.primitiveIdentity, primitiveInstanceNodes[primitiveADSIndex].primitiveInstance);

      traversalSetHitNormal(localRay, &finalHit, primitiveResources[primitiveADSIndex].vertexArray, primitiveResources[primitiveADSIndex].attributeArray, primitiveResources[primitiveADSIndex].vertexAttributeArray, primitiveResources[primitiveADSIndex].systemSettings, false);

      setHitNormal(finalHit.normal, normalize(mulMatrixVec(primitiveInstanceTransforms[primitiveADSIndex], constructFloat4(finalHit.normal, 0.f)).xyz));

#ifdef IntersectionTypeAny
      break;
#endif
    }
  }
  while (true);

#ifdef IntersectionTypeClosest
  hits[index] = finalHit;
#endif
#ifdef IntersectionTypeAny
  setHitPrimitiveIndex(hits[index].primitiveIndex, finalHit.primitiveIndex);
  setHitPrimitiveIdentity(hits[index].primitiveIdentity, finalHit.primitiveIdentity);
#endif
}

#endif
