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
@param boundingBoxes Bounding box for primitives.
@param primitiveBuffer Buffer containing primitive positions.
@param attributeBuffer Buffer containing attribute inside a structure.
@param attributePackingInfo Packing information for attribute in primitive structure.
@param primitiveBatchSize Primitives processed per thread.
@param primitiveCount Total primitives in the buffer.
@param primitiveOffsets Starting offsets and prim info for primitive buffer.
*/
Kernel void createPrimitiveBoundingBoxes(
  Device PrimitiveStruct*           finalPrimitiveArray,
  Device XAB*                       boundingBoxes,
  const Device PrimitiveStruct*     primitiveBuffer,
  const Device float*               attributeBuffer,
  constantKernelInput(PackingInfo,  attributePackingInfo),
  constantKernelInput(uint,         primitiveBatchSize),
  constantKernelInput(uint,         primitiveCount),
  constantKernelInput(uint,         primType),
  constantKernelInput(uint,         primitiveOffset)
  KERNEL_THREAD_ARGUMENTS
  KERNEL_THREADGROUP_ARGUMENTS)
{
  uint index = threadLocalIndex() + primitiveBatchSize * threadGroupIndex() * threadGroupSize();
  for (short b = 0; index < primitiveCount && b < primitiveBatchSize; index += threadGroupSize(), b++)
  {
    XAB primitiveBoundingBox;

    if (primType == PrimitiveSphere)
    {
      PrimitiveStruct outPrim  = primitiveBuffer[index];
      primitiveBoundingBox.min = outPrim.position;
      primitiveBoundingBox.max = outPrim.position;

      const float radius = extractPackedFloat(attributeBuffer, attributePackingInfo, index);

      primitiveBoundingBox.min -= constructFloat3(radius);
      primitiveBoundingBox.max += constructFloat3(radius);

      finalPrimitiveArray[index + primitiveOffset] = outPrim;
    }

    if (primType == PrimitiveTriangle)
    {
      uint3 vertIndices = constructUint3(0, 1, 2) + index * 3;

      // for indexed array a non zero stride is assumed
      if (attributePackingInfo.strideIn4Bytes > 0)
      {
        vertIndices.x = asUint(extractPackedFloat(attributeBuffer, attributePackingInfo, vertIndices.x));
        vertIndices.y = asUint(extractPackedFloat(attributeBuffer, attributePackingInfo, vertIndices.y));
        vertIndices.z = asUint(extractPackedFloat(attributeBuffer, attributePackingInfo, vertIndices.z));
      }

      // get vertex zero and vertex position
      PrimitiveStruct vert0 = primitiveBuffer[vertIndices.x];
      PrimitiveStruct vert1 = primitiveBuffer[vertIndices.y];
      const uint v1identity = vert1.identity;
      PrimitiveStruct vert2 = primitiveBuffer[vertIndices.z];
      const uint v2identity = vert2.identity;

      primitiveBoundingBox.min = min3(vert0.position, vert1.position, vert2.position);
      primitiveBoundingBox.max = max3(vert0.position, vert1.position, vert2.position);

      vert1.position = vert1.position - vert0.position;
      vert1.identity = v1identity;
      vert2.position = vert2.position - vert0.position;
      vert2.identity = v2identity;

      vertIndices = primitiveOffset + select(constructUint3(0, 1, 2) + index * 3, vertIndices, attributePackingInfo.strideIn4Bytes == 0);
      finalPrimitiveArray[vertIndices.x] = vert0;
      finalPrimitiveArray[vertIndices.y] = vert1;
      finalPrimitiveArray[vertIndices.z] = vert2;
    }

    boundingBoxes[index + primitiveOffset] = primitiveBoundingBox;
  }
}

#endif
