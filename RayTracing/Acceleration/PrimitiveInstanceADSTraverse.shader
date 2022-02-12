#ifndef PRIMITIVE_INSTANCE_ACCELERATION_DATA_STRUCT_TRAVERSE_SHADER
#define PRIMITIVE_INSTANCE_ACCELERATION_DATA_STRUCT_TRAVERSE_SHADER

Kernel void intersectRaysBVHPrimitiveInstancesFlattened(
  Device HitStruct*                   hits,
  const Device RayStruct*             rays,
  constantKernelInput(uint,           rayCount),
  const Device BVHNodeInfo*           treeInternalNodes,
  const Device uint*                  leafParentNodeIndices,
  const Device uint*                  nodeParentNodeIndices,
  const Device XAB*                   treeLeafNodeBoundingBoxes,
  const Device XAB*                   treeInternalNodeBoundingBoxes,
  const Device PrimitiveADSResources* pointers,
  const Device float4*                primitiveInstanceTransforms,
  constantKernelInput(uint,           primitiveCount),
  atomicKernelInput(uint,             rayIndexAtomicBuffer),
  sharedMemKernelInput(uint,          sharedLeafNodeIndex, 12)
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

  HitStruct hit = stacklessTraverseBinaryTree(rays[index].maxDistance,
    pointers[0].treeInternalNodes,
    pointers[0].leafParentNodeIndices,
    pointers[0].nodeParentNodeIndices,
    pointers[0].treeLeafNodeBoundingBoxes,
    pointers[0].treeInternalNodeBoundingBoxes,
    ray.origin, ray.direction, invRayDirection, sign,
    pointers[0].vertexArray,
    pointers[0].attributeArray,
    pointers[0].systemSettings,
    &primInfo, threadLocalIndex(), sharedLeafNodeIndex);

  traversalSetHitNormal(ray, &hit, pointers[0].vertexArray, pointers[0].attributeArray, pointers[0].vertexAttributeArray, pointers[0].systemSettings);
#ifdef IntersectionTypeClosest
  hits[index] = hit;
#endif
#ifdef IntersectionTypeAny
  setHitPrimitiveIndex(hits[index].primitiveIndex, hit.primitiveIndex);
  setHitPrimitiveIdentity(hits[index].primitiveIdentity, hit.primitiveIdentity);
#endif
}

Kernel void intersectRaysBVHPrimitiveInstances(
  Device HitStruct*                   hits,
  const Device RayStruct*             rays,
  constantKernelInput(uint,           rayCount),
  const Device BVHNodeInfo*           treeInternalNodes,
  const Device uint*                  leafParentNodeIndices,
  const Device uint*                  nodeParentNodeIndices,
  const Device XAB*                   treeLeafNodeBoundingBoxes,
  const Device XAB*                   treeInternalNodeBoundingBoxes,
  const Device PrimitiveADSResources* pointers,
  const Device float4*                primitiveInstanceTransforms,
  constantKernelInput(uint,           primitiveCount),
  atomicKernelInput(uint,             rayIndexAtomicBuffer),
  sharedMemKernelInput(uint,          sharedLeafNodeIndex, 12)
  KERNEL_THREAD_ARGUMENTS
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint index = threadIndex();

  if (index >= rayCount)
    return;

  HitStruct finalHit;
  initializeHit(&finalHit);

  finalHit.distance = rays[index].maxDistance;
  uint closestPrimitiveADSIndex = -1;

#if RAY_TRAVERSAL_SHARED_MEMORY_INDEX_STRIDE > 0
  Shared uint *leafNodeIndex = sharedLeafNodeIndex;
#else
  uint leafNodeIndex[RAY_TRAVERSAL_BVH_MAX_LEAFS];
#endif

  const ushort localIndex = threadLocalIndex();

  leafNodeIndex[localIndex * RAY_TRAVERSAL_SHARED_MEMORY_INDEX_STRIDE + RAY_TRAVERSAL_BVH_MAX_LEAFS] = -1;

  do
  {
    RayStruct ray = rays[index];
    float3 invRayDirection = 1.f / ray.direction;
    bool3 sign = selectInput3(invRayDirection < 0.f);

    stacklessTraverseBinaryTree(finalHit.distance, treeInternalNodes, leafParentNodeIndices, nodeParentNodeIndices, treeLeafNodeBoundingBoxes, treeInternalNodeBoundingBoxes, ray.origin, ray.direction, invRayDirection, sign, 0, 0, 0, 0, threadLocalIndex(), sharedLeafNodeIndex);

    if (leafNodeIndex[localIndex * RAY_TRAVERSAL_SHARED_MEMORY_INDEX_STRIDE + RAY_TRAVERSAL_BVH_MAX_LEAFS] == -1) break;

    const uint primitiveADSIndex = leafNodeIndex[localIndex * RAY_TRAVERSAL_SHARED_MEMORY_INDEX_STRIDE + RAY_TRAVERSAL_BVH_MAX_LEAFS + 3];
    const float4x4 transform = unpackDeviceFloat3x4To4x4(&primitiveInstanceTransforms[primitiveADSIndex * 3]);

    ray.origin    = mulMatrixVec(transform, constructFloat4(ray.origin, 1.f)).xyz;
    ray.direction = mulMatrixVec(transform, constructFloat4(ray.direction, 0.f)).xyz;
    invRayDirection = 1.f / ray.direction;

    DecodedPrimitiveInfo primInfo;
    primInfo.primitiveType = RTPrimitiveCount;

    HitStruct hit = stacklessTraverseBinaryTree(finalHit.distance,
      pointers[primitiveADSIndex].treeInternalNodes,
      pointers[primitiveADSIndex].leafParentNodeIndices,
      pointers[primitiveADSIndex].nodeParentNodeIndices,
      pointers[primitiveADSIndex].treeLeafNodeBoundingBoxes,
      pointers[primitiveADSIndex].treeInternalNodeBoundingBoxes,
      ray.origin, ray.direction, invRayDirection, sign,
      pointers[primitiveADSIndex].vertexArray,
      pointers[primitiveADSIndex].attributeArray,
      pointers[primitiveADSIndex].systemSettings,
      &primInfo, localIndex, sharedLeafNodeIndex,
      true);

    if (hit.distance < finalHit.distance)
    {
      finalHit = hit;
      closestPrimitiveADSIndex = primitiveADSIndex;

      traversalSetHitNormal(ray, &finalHit, pointers[closestPrimitiveADSIndex].vertexArray, pointers[closestPrimitiveADSIndex].attributeArray, pointers[closestPrimitiveADSIndex].vertexAttributeArray, pointers[closestPrimitiveADSIndex].systemSettings);
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
