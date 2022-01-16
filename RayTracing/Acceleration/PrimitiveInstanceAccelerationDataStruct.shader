#ifndef PRIMITIVE_INSTANCE_ACCELERATION_DATA_STRUCT_SHADER
#define PRIMITIVE_INSTANCE_ACCELERATION_DATA_STRUCT_SHADER

/*
@kernel Update per instance data based on instance transform.
@param primitiveInstanceNodes Primitive instance node data.
@param primitiveInstanceTransforms Primitive instance transforms and inverse transforms.
@param primitiveADSTreeInternalNodeBoundingBoxes Primitive acceleration data structure.
@param primitiveInstanceCount Total primitives in the buffer.
*/
Kernel void updatePrimitiveInstanceData(
  Device PrimitiveInstanceADSLeaf*    primitiveInstanceNodes,
  Device float4*                      primitiveInstanceTransforms,
  const Device RayTracingPointerType* primitiveADSTreeInternalNodeBoundingBoxes,
  constantKernelInput(uint,           primitiveInstanceCount)
  KERNEL_THREAD_ARGUMENTS
  KERNEL_THREADGROUP_ARGUMENTS)
{
  const uint index = threadLocalIndex();

  if (index >= primitiveInstanceCount) return;

  PrimitiveInstanceADSLeaf leafNode           = primitiveInstanceNodes[index];
  const uint primitiveADSIndex                = leafNode.primitiveADSIndex;
  const RayTracingEntityId primitiveInstance  = leafNode.primitiveInstance;

  const uint instanceIndex        = getRayTracingInstanceId(primitiveInstance);
  const XAB primitiveBoundingBox  = ((const Device XAB*)primitiveADSTreeInternalNodeBoundingBoxes[primitiveADSIndex].pointer)[0];
  const float4x4 transform        = unpackDeviceFloat3x4To4x4(&primitiveInstanceTransforms[index * 3]);

  float3 newMin = constructFloat3(INFINITY);
  float3 newMax = constructFloat3(-INFINITY);

  for (ushort i=0; i<2; i++)
  {
    for (ushort j=0; j<2; j++)
    {
      for (ushort k=0; k<2; k++)
      {
        float3 position = select(primitiveBoundingBox.min, primitiveBoundingBox.max, selectInput3(constructUshort3(i, j, k) > 0));
        position = mulMatrixVec(transform, constructFloat4(position, 1.f)).xyz;
        newMin = min(newMin, position);
        newMax = max(newMax, position);
      }
    }
  }

  leafNode.bounds.min = newMin;
  leafNode.bounds.max = newMax;
  leafNode.primitiveADSIndex = primitiveADSIndex;
  leafNode.primitiveInstance = primitiveInstance;

  primitiveInstanceNodes[index] = leafNode;

  float4x4 invTransform = transpose(transform) / determinant(transform);
  (((Thread float4*)&invTransform)[0]).w = (((Thread float4*)&invTransform)[3]).x;
  (((Thread float4*)&invTransform)[1]).w = (((Thread float4*)&invTransform)[3]).y;
  (((Thread float4*)&invTransform)[2]).w = (((Thread float4*)&invTransform)[3]).z;

  packFloat4x4ToDevice3x4(invTransform, &primitiveInstanceTransforms[(primitiveInstanceCount + index) * 3])
}

/*
Kernel void intersectRaysBVH(
  Device HitStruct*                   hits,
  const Device RayStruct*             rays,
  constantKernelInput(uint,           rayCount),
  const Device RayTracingPointerType* pointersVertexArray,
  const Device RayTracingPointerType* pointersAttributeArray,
  const Device RayTracingPointerType* pointersTreeInternalNodes,
  const Device RayTracingPointerType* pointersLeafParentNodeIndices,
  const Device RayTracingPointerType* pointersNodeParentNodeIndices,
  const Device RayTracingPointerType* pointersTreeLeafNodeBoundingBoxes,
  const Device RayTracingPointerType* pointersTreeInternalNodeBoundingBoxes,
  Const RayTracingConstPointerType*   pointersSystemSettings,
  constantKernelInput(uint,           primitiveCount),
  atomicKernelInput(uint,             rayIndexAtomicBuffer),
  sharedMemKernelInput(uint,          sharedLeafNodeIndex, 13)
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

  //HitStruct hit = stackTraverseBinaryTree(rays[index].maxDistance, treeInternalNodes, leafParentNodeIndices, nodeParentNodeIndices, treeInternalNodeBoundingBoxes, rayOrigin, rayDirection, invRayDirection, sign, vertexArray, attributeArray, systemSettings);
  //HitStruct hit = stacklessTraverseBinaryTree(rays[index].maxDistance, treeInternalNodes, leafParentNodeIndices, nodeParentNodeIndices, treeLeafNodeBoundingBoxes, treeInternalNodeBoundingBoxes, rayOrigin, rayDirection, invRayDirection, sign, vertexArray, attributeArray, systemSettings, &primInfo, threadLocalIndex(), sharedLeafNodeIndex);

  HitStruct hit = stacklessTraverseBinaryTree(rays[index].maxDistance,
    (const Device BVHNodeInfo*)pointersTreeInternalNodes[0].pointer,
    (const Device uint*)pointersLeafParentNodeIndices[0].pointer,
    (const Device uint*)pointersNodeParentNodeIndices[0].pointer,
    (const Device XAB*)pointersTreeLeafNodeBoundingBoxes[0].pointer,
    (const Device XAB*)pointersTreeInternalNodeBoundingBoxes[0].pointer,
    rayOrigin, rayDirection, invRayDirection, sign,
    (const Device PrimitiveStruct*)pointersVertexArray[0].pointer,
    (const Device PrimitiveAttrib*)pointersAttributeArray[0].pointer,
    (Const RTSystemSettings*)pointersSystemSettings[0].pointer,
    &primInfo, threadLocalIndex(), sharedLeafNodeIndex);

#ifdef IntersectionTypeClosest
  hits[index] = hit;
#endif
#ifdef IntersectionTypeAny
  setHitPrimitiveIndex(hits[index].primitiveIndex, hit.primitiveIndex);
  setHitPrimitiveIdentity(hits[index].primitiveIdentity, hit.primitiveIdentity);
#endif
}*/

#endif
