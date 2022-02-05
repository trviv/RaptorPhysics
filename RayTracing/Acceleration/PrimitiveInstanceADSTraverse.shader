#ifndef PRIMITIVE_INSTANCE_ACCELERATION_DATA_STRUCT_TRAVERSE_SHADER
#define PRIMITIVE_INSTANCE_ACCELERATION_DATA_STRUCT_TRAVERSE_SHADER

Kernel void intersectRaysBVHPrimitiveInstances(
  Device HitStruct*                   hits,
  const Device RayStruct*             rays,
  constantKernelInput(uint,           rayCount),
  const Device PrimitiveADSResources*    pointers,
  constantKernelInput(uint,           primitiveCount),
  atomicKernelInput(uint,             rayIndexAtomicBuffer),
  sharedMemKernelInput(uint,          sharedLeafNodeIndex, 6)
  KERNEL_THREAD_ARGUMENTS
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
  primInfo.primitiveType = RTPrimitiveCount;

  HitStruct hit = stacklessTraverseBinaryTree(rays[index].maxDistance,
    pointers[0].treeInternalNodes,
    pointers[0].leafParentNodeIndices,
    pointers[0].nodeParentNodeIndices,
    pointers[0].treeLeafNodeBoundingBoxes,
    pointers[0].treeInternalNodeBoundingBoxes,
    rayOrigin, rayDirection, invRayDirection, sign,
    pointers[0].vertexArray,
    pointers[0].attributeArray,
    pointers[0].systemSettings,
    &primInfo, threadLocalIndex(), sharedLeafNodeIndex);

#ifdef IntersectionTypeClosest
  hits[index] = hit;
#endif
#ifdef IntersectionTypeAny
  setHitPrimitiveIndex(hits[index].primitiveIndex, hit.primitiveIndex);
  setHitPrimitiveIdentity(hits[index].primitiveIdentity, hit.primitiveIdentity);
#endif
}

#endif
