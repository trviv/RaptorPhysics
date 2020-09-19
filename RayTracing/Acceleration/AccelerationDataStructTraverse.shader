#ifndef ACCELERATION_DATA_STRUCT_TRAVERSE_SHADER
#define ACCELERATION_DATA_STRUCT_TRAVERSE_SHADER

/*
@kernel Calculate bounding box for individual primitives.
@param boundingBoxes Bounding box for the group.
@param primitiveBuffer Buffer containing primitive positions.
@param radiusBuffer Buffer containing radius inside a structure.
@param radiusPackingInfo Packing information for position in primitive structure.
@param primitiveBatchSize Primitives processed per thread.
@param primitiveCount Total primitives in the buffer.
@param primitiveOffset Starting offset for the bounding box output.
*/
Kernel void intersectRays(
  Device HitStruct*                 hits,
  const Device RayStruct*           rays,
  constantKernelInput(uint,         rayCount),
  const Device XAB*                 boundingBoxes,
  const Device PrimitiveStruct*     primitiveBuffer,
  const Device float*               attributeBuffer,
  constantKernelInput(PackingInfo,  attributePackingInfo),
  constantKernelInput(uint,         primitiveCount)
  KERNEL_GLOBAL_ARGUMENTS)
{
  uint index = threadIndex();

  if (index >= rayCount)
    return;

  for (uint primIndex = 0; primIndex < primitiveCount; primIndex++)
  {
    //if (boundingBoxes[primIndex])
  }
}

#endif
