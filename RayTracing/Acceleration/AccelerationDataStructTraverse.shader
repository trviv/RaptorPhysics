#ifndef ACCELERATION_DATA_STRUCT_TRAVERSE_SHADER
#define ACCELERATION_DATA_STRUCT_TRAVERSE_SHADER

/*
@kernel Intersect rays with primitives and fill hit info.
@param hits Hit info buffer.
@param rays Ray buffer.
@param rayCount Ray count.
@param boundingBoxes Bounding box for primitives.
@param primitiveBuffer Buffer containing primitive positions.
@param attributeBuffer Buffer containing attribute inside a structure.
@param attributePackingInfo Packing information for attribute in primitive structure.
@param primitiveCount Total primitives in the buffer.
@param primitiveOffsets Starting offsets and prim info for primitive buffer.
*/
Kernel void intersectRays(
  Device HitStruct*                 hits,
  const Device RayStruct*           rays,
  constantKernelInput(uint,         rayCount),
  const Device XAB*                 boundingBoxes,
  const Device PrimitiveStruct*     primitiveBuffer,
  const Device float*               attributeBuffer,
  constantKernelInput(PackingInfo,  attributePackingInfo),
  constantKernelInput(uint,         primitiveCount),
  Const RTPrimitiveOffset*          primitiveOffsets
  KERNEL_GLOBAL_ARGUMENTS)
{
  uint index = threadIndex();

  if (index >= rayCount)
    return;

  const float3 invRayDirection = 1.f / rays[index].direction;
  const bool3 sign = invRayDirection < 0.f;
  const float3 rayOrigin = rays[index].origin;

  HitStruct hit;
  hit.distance = INFINITY;
  hit.primitiveIndex = -1;

  for (uint primIndex = 0; primIndex < primitiveCount; primIndex++)
  {
    if (rayXABIntersectEarliest(&hit.distance, boundingBoxes[primIndex], rayOrigin, invRayDirection, sign))
    {
      hit.primitiveIndex = primIndex;
    }
  }

  hits[index] = hit;
}

#endif
