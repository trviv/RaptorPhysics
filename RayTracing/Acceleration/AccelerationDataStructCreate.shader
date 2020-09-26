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

inline XAB createPrimitiveXAB(
  const ushort                      primType,
  const uint                        index,
  const Device PrimitiveStruct*     primitiveBuffer,
  const Device float*               attributeBuffer,
  constantKernelInput(PackingInfo,  attributePackingInfo))
{
  const float3 position = primitiveBuffer[index].position;
  const float radius    = extractPackedFloat(attributeBuffer, attributePackingInfo, index);

  XAB primitiveBoundingBox;
  primitiveBoundingBox.min = position;
  primitiveBoundingBox.max = position;

  if (primType == PrimitiveSphere)
  {
    primitiveBoundingBox.min -= constructFloat3(radius);
    primitiveBoundingBox.max += constructFloat3(radius);
  }

  return primitiveBoundingBox;
}

/*
@kernel Calculate bounding box for individual spheres.
@param boundingBoxes Bounding box for primitives.
@param primitiveBuffer Buffer containing primitive positions.
@param attributeBuffer Buffer containing attribute inside a structure.
@param attributePackingInfo Packing information for attribute in primitive structure.
@param primitiveBatchSize Primitives processed per thread.
@param primitiveCount Total primitives in the buffer.
@param primitiveOffsets Starting offsets and prim info for primitive buffer.
*/
Kernel void createPrimitiveBoundingBoxes(
  Device XAB*                       boundingBoxes,
  const Device PrimitiveStruct*     primitiveBuffer,
  const Device float*               attributeBuffer,
  constantKernelInput(PackingInfo,  attributePackingInfo),
  constantKernelInput(uint,         primitiveBatchSize),
  constantKernelInput(uint,         primitiveCount),
  Const RTPrimitiveOffset*          primitiveOffsets
  KERNEL_THREAD_ARGUMENTS
  KERNEL_THREADGROUP_ARGUMENTS)
{
  uint index = threadLocalIndex() + primitiveBatchSize * threadGroupIndex() * threadGroupSize();
  for (short b = 0; index < primitiveCount && b < primitiveBatchSize; index += threadGroupSize(), b++)
  {
    const ushort primType = getSystemPrimType(primitiveOffsets, index);
    boundingBoxes[index] = createPrimitiveXAB(primType, index, primitiveBuffer, attributeBuffer, attributePackingInfo);
  }
}

#endif
