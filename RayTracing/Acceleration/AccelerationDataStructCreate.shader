#ifndef ACCELERATION_DATA_STRUCT_CREATE_SHADER
#define ACCELERATION_DATA_STRUCT_CREATE_SHADER

float3 extractPackedFloat3(const Device float* buffer, const PackingInfo packingInfo, const uint index)
{
  return *((Device float3*)(buffer + index * packingInfo.strideIn4Bytes + packingInfo.offsetIn4Bytes));
}

float extractPackedFloat(const Device float* buffer, const PackingInfo packingInfo, const uint index)
{
  return buffer[index * packingInfo.strideIn4Bytes + packingInfo.offsetIn4Bytes];
}

/*
@kernel Calculate bounding box for individual spheres.
@param boundingBoxes Bounding box for the group.
@param primitiveBuffer Buffer containing primitive positions.
@param radiusBuffer Buffer containing radius inside a structure.
@param radiusPackingInfo Packing information for position in primitive structure.
@param primitiveBatchSize Primitives processed per thread.
@param primitiveCount Total primitives in the buffer.
@param primitiveOffset Starting offset for the bounding box output.
*/
Kernel void createPrimitiveBoundingBoxes(
  Device XAB*                       boundingBoxes,
  const Device PrimitiveStruct*     primitiveBuffer,
  const Device float*               radiusBuffer,
  constantKernelInput(PackingInfo,  radiusPackingInfo),
  constantKernelInput(uint,         primitiveBatchSize),
  constantKernelInput(uint,         primitiveCount),
  constantKernelInput(uint,         primitiveOffset)
  KERNEL_THREAD_ARGUMENTS
  KERNEL_THREADGROUP_ARGUMENTS)
{
  uint index = threadLocalIndex() + primitiveBatchSize * threadGroupIndex() * threadGroupSize();
  for (short b = 0; index < primitiveCount && b < primitiveBatchSize; index += threadGroupSize(), b++)
  {
    const float3 position = primitiveBuffer[index].position;
    const float radius    = extractPackedFloat(radiusBuffer, radiusPackingInfo, index);

    XAB primitiveBoundingBox;
    primitiveBoundingBox.min = position - constructFloat3(radius);
    primitiveBoundingBox.max = position + constructFloat3(radius);

    boundingBoxes[primitiveOffset+index] = primitiveBoundingBox;
  }
}

#endif
