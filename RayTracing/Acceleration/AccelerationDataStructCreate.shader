#ifndef ACCELERATION_DATA_STRUCT_CREATE_SHADER
#define ACCELERATION_DATA_STRUCT_CREATE_SHADER

#include "ComputeHeader.shader"
#include "ComputeShared.h"
#include "RayTracingStruct.h"

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
  DecodedPrimitiveInfo primInfo = defaultPrimitiveInfo();

  uint index = threadLocalIndex() + primitiveBatchSize * threadGroupIndex() * threadGroupSize();
  for (short b = 0; index < primitiveCount && b < primitiveBatchSize; index += threadGroupSize(), b++)
  {
    XAB primitiveBoundingBox;

    decodePrimitiveInfoFromSystemSettings(systemSettings, index, &primInfo);

    if (primInfo.primitiveType == PrimitiveSphere)
    {
      const PrimitiveStruct sphere = vertexArray[primInfo.vertexOffset + index - primInfo.primitiveOffset];
      const float radius = attributeArray[index].radius;

      primitiveBoundingBox.min = sphere.position - radius;
      primitiveBoundingBox.max = sphere.position + radius;
    }
    else
    {
      uint4 quadIndex;
      if (primInfo.primitiveType == PrimitiveIndexedTriangle)
      {
        quadIndex = constructUint4(attributeArray[index].triangleIndex, -1);
      }
      else if (primInfo.primitiveType == PrimitiveTriangle)
      {
        quadIndex = constructUint4(primInfo.vertexOffset + (index - primInfo.primitiveOffset) * 3 + constructUint3(0, 1, 2), -1);
      }
      else if (primInfo.primitiveType == PrimitiveIndexedQuad)
      {
        quadIndex = attributeArray[index].quadIndex;
      }

      const float3 vert0 = vertexArray[quadIndex.x].position;
      float3 vert1 = vertexArray[quadIndex.y].position;
      float3 vert2 = vertexArray[quadIndex.z].position;

      if (primInfo.primitiveType == PrimitiveTriangle)
      {
        vert1 += vert0;
        vert2 += vert0;
      }

      primitiveBoundingBox.min = min3(vert0, vert1, vert2);
      primitiveBoundingBox.max = max3(vert0, vert1, vert2);

      if (quadIndex.w != -1)
      {
        const float3 vert3 = vertexArray[quadIndex.w].position;

        primitiveBoundingBox.min = min(primitiveBoundingBox.min, vert3);
        primitiveBoundingBox.max = max(primitiveBoundingBox.max, vert3);
      }
    }

    primitiveBoundingBox.min -= MIN_TIME;
    primitiveBoundingBox.max += MIN_TIME;
    boundingBoxes[index] = primitiveBoundingBox;
  }
}

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
  const Device PrimitiveAttrib* attributeArray,
  Const RTSystemSettings*       systemSettings,
  constantKernelInput(uint,     primitiveBatchSize),
  constantKernelInput(uint,     primitiveCount)
  KERNEL_THREAD_ARGUMENTS
  KERNEL_THREADGROUP_ARGUMENTS)
{
  DecodedPrimitiveInfo primInfo = defaultPrimitiveInfo();

  const float3 inverseMergedBoxSize = 1024.f / (systemSettings->systemBound.max - systemSettings->systemBound.min);
  const float3 mergedBoxCenter = (systemSettings->systemBound.min + systemSettings->systemBound.max) * 0.5f;

  uint index = threadLocalIndex() + primitiveBatchSize * threadGroupIndex() * threadGroupSize();
  for (short b = 0; index < primitiveCount && b < primitiveBatchSize; index += threadGroupSize(), b++)
  {
    decodePrimitiveInfoFromSystemSettings(systemSettings, index, &primInfo);
    float3 center;

    if (primInfo.primitiveType == PrimitiveSphere)
    {
      center = vertexArray[primInfo.vertexOffset + (index - primInfo.primitiveOffset)].position;
    }
    else
    {
      uint4 quadIndex;
      if (primInfo.primitiveType == PrimitiveIndexedTriangle)
      {
        quadIndex = constructUint4(attributeArray[index].triangleIndex, -1);
      }
      else if (primInfo.primitiveType == PrimitiveTriangle)
      {
        quadIndex = constructUint4(primInfo.vertexOffset + (index - primInfo.primitiveOffset) * 3 + constructUint3(0, 1, 2), -1);
      }
      else if (primInfo.primitiveType == PrimitiveIndexedQuad)
      {
        quadIndex = attributeArray[index].quadIndex;
      }

      const float3 vert0 = vertexArray[quadIndex.x].position;
      const float3 vert1 = vertexArray[quadIndex.y].position;
      const float3 vert2 = vertexArray[quadIndex.z].position;

      center = (vert0 + vert1 + vert2);

      if (primInfo.primitiveType == PrimitiveTriangle)
      {
        center += 2.f * vert0;
      }

      if (quadIndex.w != -1)
      {
        const float3 vert3 = vertexArray[quadIndex.w].position;
        center = (center + vert3) * 1.f/4.f;
      }
      else
      {
        center = center * 1.f/3.f;
      }
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
