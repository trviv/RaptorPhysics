#ifndef ACCELERATION_DATA_STRUCT_CREATE_SHADER
#define ACCELERATION_DATA_STRUCT_CREATE_SHADER

/*
@kernel Calculate bounding box for individual primitives.
@param boundingBoxes Bounding box for primitives.
@param vertexArray Buffer containing all positions.
@param attributeArray Buffer containing primitive attribute data.
@param systemSettings Settings for the ray tracing system.
@param primitiveBatchSize Primitives processed per thread.
@param primitiveCount Total primitives in the buffer.
*/
Kernel void createPrimitiveBoundingBoxes(
  Device XAB*                   boundingBoxes,
  const Device PrimitiveStruct* vertexArray,
  const Device PrimitiveAttrib* attributeArray,
  Const RTSystemSettings*       systemSettings,
  constantKernelInput(uint,     primitiveBatchSize),
  constantKernelInput(uint,     primitiveCount)
  KERNEL_THREAD_ARGUMENTS
  KERNEL_THREADGROUP_ARGUMENTS)
{
  uint index = threadLocalIndex() + primitiveBatchSize * threadGroupIndex() * threadGroupSize();
  for (short b = 0; index < primitiveCount && b < primitiveBatchSize; index += threadGroupSize(), b++)
  {
    XAB primitiveBoundingBox;

    const DecodedPrimitiveInfo primInfo = decodePrimitiveInfoFromSystemSettings(systemSettings, index);

    if (primInfo.primType == PrimitiveSphere)
    {
      const PrimitiveStruct sphere = vertexArray[primInfo.vertexOffset + index - primInfo.indexOffset];
      const float radius = attributeArray[index].radius;

      primitiveBoundingBox.min = sphere.position - radius;
      primitiveBoundingBox.max = sphere.position + radius;
    }

    if (primInfo.primType == PrimitiveTriangle)
    {
      const uint triIndex = primInfo.vertexOffset + (index - primInfo.indexOffset)*3;

      const float3 vert0 = vertexArray[triIndex].position;
      const float3 vert1 = vertexArray[triIndex+1].position + vert0;
      const float3 vert2 = vertexArray[triIndex+2].position + vert0;

      primitiveBoundingBox.min = min3(vert0, vert1, vert2);
      primitiveBoundingBox.max = max3(vert0, vert1, vert2);
    }

    boundingBoxes[index] = primitiveBoundingBox;
  }
}

// TODO: Remove if not want to keep it around
/*Kernel void createPrimitiveBoundingBoxesLegacy(
  Device PrimitiveStruct*           finalVertexArray,
  Device PrimitiveAttrib*           finalAttributeArray,
  Device XAB*                       boundingBoxes,
  const Device PrimitiveStruct*     primitiveBuffer,
  constantKernelInput(PackingInfo,  primitivePackingInfo),
  const Device float*               attributeBuffer,
  constantKernelInput(PackingInfo,  attributePackingInfo),
  constantKernelInput(uint,         primitiveBatchSize),
  constantKernelInput(uint,         primitiveCount),
  constantKernelInput(uint,         primType),
  constantKernelInput(uint,         primitiveOffset),
  constantKernelInput(uint,         vertexOffset)
  KERNEL_THREAD_ARGUMENTS
  KERNEL_THREADGROUP_ARGUMENTS)
{
  uint index = threadLocalIndex() + primitiveBatchSize * threadGroupIndex() * threadGroupSize();
  for (short b = 0; index < primitiveCount && b < primitiveBatchSize; index += threadGroupSize(), b++)
  {
    XAB primitiveBoundingBox;

    if (primType == PrimitiveSphere)
    {
      PrimitiveStruct outPrim  = primitiveBuffer[index + primitivePackingInfo.elementOffset];
      primitiveBoundingBox.min = outPrim.position;
      primitiveBoundingBox.max = outPrim.position;

      const float radius = extractPackedFloat(attributeBuffer, attributePackingInfo, index + primitivePackingInfo.elementOffset);

      primitiveBoundingBox.min -= constructFloat3(radius);
      primitiveBoundingBox.max += constructFloat3(radius);

      finalVertexArray[index + vertexOffset] = outPrim;
      finalAttributeArray[index + vertexOffset].radius = radius;
    }

    if (primType == PrimitiveTriangle)
    {
      uint3 vertIndices = constructUint3(0, 1, 2) + index * 3 + primitivePackingInfo.elementOffset;

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
      const IdentityInfo v1identity = vert1.identity;
      PrimitiveStruct vert2 = primitiveBuffer[vertIndices.z];
      const IdentityInfo v2identity = vert2.identity;

      primitiveBoundingBox.min = min3(vert0.position, vert1.position, vert2.position);
      primitiveBoundingBox.max = max3(vert0.position, vert1.position, vert2.position);

      vert1.position = vert1.position - vert0.position;
      vert1.identity = v1identity;
      vert2.position = vert2.position - vert0.position;
      vert2.identity = v2identity;

      vertIndices = vertexOffset + select(constructUint3(0, 1, 2) + index * 3, vertIndices, selectInput3(attributePackingInfo.strideIn4Bytes == 0));
      finalVertexArray[vertIndices.x] = vert0;
      finalVertexArray[vertIndices.y] = vert1;
      finalVertexArray[vertIndices.z] = vert2;
    }

    boundingBoxes[index + primitiveOffset] = primitiveBoundingBox;
  }
}*/

/*
@kernel Compute and store morton code for each primitive.
@param bvhLeafs Particle position and index data for bounding volume hierarchy.
@param vertexArray Primitive position array.
@param systemSettings Settings for the ray tracing system.
@param primitiveBatchSize Primitives processed per thread.
@param primitiveCount Total primitives in the buffer.
*/
Kernel void assignMortonCode(
  Device BVHLeafInfo*           bvhLeafs,
  const Device PrimitiveStruct* vertexArray,
  Const RTSystemSettings*       systemSettings,
  constantKernelInput(uint,     primitiveBatchSize),
  constantKernelInput(uint,     primitiveCount)
  KERNEL_THREAD_ARGUMENTS
  KERNEL_THREADGROUP_ARGUMENTS)
{
  const float3 inverseMergedBoxSize = 1024.f / (systemSettings->systemBound.max - systemSettings->systemBound.min);
  const float3 mergedBoxCenter = (systemSettings->systemBound.min + systemSettings->systemBound.max) * 0.5f;

  uint index = threadLocalIndex() + primitiveBatchSize * threadGroupIndex() * threadGroupSize();
  for (short b = 0; index < primitiveCount && b < primitiveBatchSize; index += threadGroupSize(), b++)
  {
    const DecodedPrimitiveInfo primInfo = decodePrimitiveInfoFromSystemSettings(systemSettings, index);
    float3 center;

    if (primInfo.primType == PrimitiveSphere)
    {
      center = vertexArray[primInfo.vertexOffset + (index - primInfo.indexOffset)].position;
    }

    if (primInfo.primType == PrimitiveTriangle)
    {
      const uint triIndex = primInfo.vertexOffset + (index - primInfo.indexOffset)*3;

      const float3 vert0 = vertexArray[triIndex].position;
      const float3 edge1 = vertexArray[triIndex+1].position;
      const float3 edge2 = vertexArray[triIndex+2].position;

      center = vert0 + (edge1 + edge2) * 0.333f;
    }

    // Quantize into integer coordinates
    // floor() is needed to prevent the center cell, at (0,0,0) from being twice the size
    float3 positionRelativeToCenter = (center - mergedBoxCenter) * inverseMergedBoxSize;

    int3 quantizedPosition = convertInt3(select(floor(positionRelativeToCenter), positionRelativeToCenter, positionRelativeToCenter >= 0.0f));

    // Clamp coordinates into [-512, 511], then convert range from [-512, 511] to [0, 1023]
    quantizedPosition = max(constructInt3(-512), min(quantizedPosition, constructInt3(511))) + constructInt3(512);

    //Interleave bits(assign a morton code, also known as a z-curve)
    BVHLeafInfo bvhLeaf;
    bvhLeaf.mortonCode = encode32BitMortonCode(quantizedPosition);
    bvhLeaf.index = index;

    bvhLeafs[index] = bvhLeaf;
  }
}

#endif
