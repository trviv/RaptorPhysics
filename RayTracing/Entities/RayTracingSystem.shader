#ifndef RAY_TRACING_SYSTEM_SHADER
#define RAY_TRACING_SYSTEM_SHADER

/*
@kernel Create single array composed of all the primitives.
@param finalVertexArray Buffer containing all positions.
@param finalAttributeArray Buffer containing primitive attribute data.
@param primitiveBuffer Buffer containing primitive positions.
@param attributeBuffer Buffer containing attribute inside a structure.
@param attributePackingInfo Packing information for attribute in primitive structure.
@param primitiveBatchSize Primitives processed per thread.
@param primitiveCount Total primitives in the buffer.
@param primType Primitive type for the dispatch.
@param primitiveOffset Starting offset for storing primitive data.
@param vertexOffset Starting offset for storing vertex data.
*/
Kernel void collectPrimitives(
  Device PrimitiveStruct*           finalVertexArray,
  Device PrimitiveAttrib*           finalAttributeArray,
  const Device PrimitiveStruct*     primitiveBuffer,
  constantKernelInput(PackingInfo,  primitivePackingInfo),
  const Device float*               attributeBuffer,
  constantKernelInput(PackingInfo,  attributePackingInfo),
  constantKernelInput(IdentityInfo, primitiveIdentity),
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
    if (primType == PrimitiveSphere)
    {
      PrimitiveStruct outPrim  = primitiveBuffer[index + primitivePackingInfo.elementOffset];
      const float radius = extractPackedFloat(attributeBuffer, attributePackingInfo, index + attributePackingInfo.elementOffset);

      outPrim.identity = primitiveIdentity;
      finalVertexArray[index + vertexOffset] = outPrim;
      finalAttributeArray[index + vertexOffset].radius = radius;
    }

    if (primType == PrimitiveTriangle)
    {
      uint3 vertIndices = constructUint3(0, 1, 2) + index * 3 + attributePackingInfo.elementOffset;

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
      //const IdentityInfo v1identity = vert1.identity;
      PrimitiveStruct vert2 = primitiveBuffer[vertIndices.z];
      //const IdentityInfo v2identity = vert2.identity;

      vert0.identity = primitiveIdentity;
      vert1.position = vert1.position - vert0.position;
      //vert1.identity = v1identity;
      vert1.identity = primitiveIdentity;
      vert2.position = vert2.position - vert0.position;
      //vert2.identity = v2identity;
      vert2.identity = primitiveIdentity;

      vertIndices = vertexOffset + constructUint3(0, 1, 2) + index * 3;
      finalVertexArray[vertIndices.x] = vert0;
      finalVertexArray[vertIndices.y] = vert1;
      finalVertexArray[vertIndices.z] = vert2;
    }
  }
}

/*
@kernel Shade ray intersection in a surface based on its material properties and visiblity info.
@param colorOut Final color output.
@param shadowRays Shadow ray buffer.
@param rays Ray buffer.
@param hits Hit info buffer.
@param rayCount Ray count.
*/
Kernel void shadeIntersection(
  Device colorType4*            colorOut,
  Device RayStruct*             shadowRays,
  Device RayStruct*             rays,
  const Device HitStruct*       hits,
  constantKernelInput(uint,     rayCount),
  Const LightStruct*            lights,
  constantKernelInput(ushort,   lightOffset),
  constantKernelInput(ushort,   lightCount),
  const Device MaterialStruct*  materials
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint index = threadIndex();

  if (index >= rayCount)
    return;

#if defined(RayStructColor) && !defined(HitStructIndex)
  HitStruct hit = hits[index];
  RayStruct ray = rays[index];
  RayStruct shadowRay;
  RayStruct childRay;

  const MaterialId materialId = hit.primitiveIdentity;
  MaterialStruct material;
  ushort materialType = -1;

  if (hit.primitiveIndex != -1)
  {
    shadowRay.origin = ray.origin + ray.direction * hit.distance;
    material = materials[materialId.identity & 0x7FFFFFFF];
    materialType = getMaterialType(material);

#ifdef HitStructNormal
    if (materialId.identity > 0x7FFFFFFF && dot(ray.direction, hit.normal) >= 0.f)
    {
      hit.normal = -hit.normal;
    }
#endif

    colorType4 finalColor = colorOut[ray.rayIndex];
    finalColor.xyz += material.emissive.xyz * ray.color.xyz;
    finalColor.w = 1.f;
    colorOut[ray.rayIndex] = finalColor;
  }

  if (materialType == MaterialTypeReflective)
  {
    childRay = ray;
    // produce child ray if needed
    childRay.origin    = shadowRay.origin;
#ifdef HitStructNormal
    childRay.direction = reflectVector(ray.direction, hit.normal);
#endif
    childRay.maxDistance = INFINITY;
    rays[index] = childRay;
  }
  else if (materialType == MaterialTypeTranslucent)
  {
    childRay = ray;
    // produce child ray if needed
    childRay.origin    = shadowRay.origin;
#ifdef HitStructNormal
    childRay.direction = refractVector(ray.direction, hit.normal, getMaterialRefractiveIndex(material));
#endif
    childRay.maxDistance = INFINITY;
    rays[index] = childRay;
  }
  else
  {
    rays[index].maxDistance = 0.f;
  }

  for (ushort i=lightOffset; i<lightCount; i++)
  {
    shadowRay.color = constructColor4(0.f);
    shadowRay.maxDistance = 0.f;
    if (materialType == MaterialTypePlastic)
    {
      float3 lightPosition, lightColor;
      sampleLight(&lightPosition, &lightColor, lights[i]);
      float3 direction = lightPosition - shadowRay.origin;

#ifdef HitStructNormal
      if (dot(direction, hit.normal) >= 0.f)
#endif
      {
        const float maxDistance = length(direction);
        direction /= maxDistance;
        shadowRay.direction = direction;
        shadowRay.maxDistance = maxDistance;
        shadowRay.rayIndex  = ray.rayIndex;
        shadowRay.color.xyz = constructColor3(lightColor.xyz) * ray.color.xyz * shadeMaterialAtIntersection(material, direction, ray.direction, hit).xyz;
      }
      shadowRays[index + rayCount * i] = shadowRay;
    }
    else
    {
      shadowRays[index + rayCount * i].maxDistance = 0.f;
    }
  }
#endif
}

#define RAYS_REARRANGE_MULTIPLIER 2

inline void bitonicSortSharedUint2(
  Shared uint2* localNodes,
  const short   maxDepth,
  const short   localThreadCount,
  const short   localIndex)
{
  localMemBarrier();

  for (short mergeSize = 2; mergeSize <= (1 << maxDepth); mergeSize <<= 1)
  {
    for (short mergeSubSize = mergeSize>>1; mergeSubSize > 0; mergeSubSize >>= 1)
    {
      for (short m=0; m<RAYS_REARRANGE_MULTIPLIER; m++)
      {
        const short indexLow  = (localIndex + localThreadCount * m) & (mergeSubSize - 1);
        const short indexHigh = (localIndex + localThreadCount * m - indexLow) << 1;
        const short index     = indexHigh + indexLow;
        const short swapIndex = indexHigh + select(mergeSubSize + indexLow, 2 * mergeSubSize - 1 - indexLow, mergeSubSize == (mergeSize >> 1));

        if (swapIndex < localThreadCount * RAYS_REARRANGE_MULTIPLIER && index < localThreadCount * RAYS_REARRANGE_MULTIPLIER)
        {
          const uint2 node1 = localNodes[index].xy;
          const uint2 node2 = localNodes[swapIndex].xy;

          if (node1.x > node2.x)
          {
            localNodes[index].xy     = node2;
            localNodes[swapIndex].xy = node1;
          }
        }
      }
      localMemBarrier();
    }
  }
}

Kernel void reorderRays(
  Device RayStruct*           rays,
  constantKernelInput(uint,   rayCount),
  sharedMemKernelInput(uint2, raySpatialData, 2)
  KERNEL_GLOBAL_ARGUMENTS
  KERNEL_THREAD_ARGUMENTS
  KERNEL_THREADGROUP_ARGUMENTS)
{
  RayStruct localRays[RAYS_REARRANGE_MULTIPLIER];

  for (short i=0; i<RAYS_REARRANGE_MULTIPLIER; i++)
  {
    uint2 rayData = constructUint2(-1);
    const uint threadGlobalIndex = threadLocalIndex() + threadGroupSize() * (i + threadGroupIndex() * RAYS_REARRANGE_MULTIPLIER);
    if (threadGlobalIndex < rayCount)
    {
      localRays[i] = rays[threadGlobalIndex];
      const uint cellInternalSpatialIndex = encode32BitMortonCode(constructInt3(511.f * (localRays[i].direction + 1.f)));
      rayData = constructUint2(cellInternalSpatialIndex, threadGlobalIndex);
    }
    raySpatialData[threadLocalIndex() + i * threadGroupSize()] = rayData;
  }

  const short tgSizePowOf2 = 32 - clz((int)threadGroupSize() * RAYS_REARRANGE_MULTIPLIER) - 1;
  bitonicSortSharedUint2(raySpatialData, tgSizePowOf2 / 2, threadGroupSize(), threadLocalIndex());

  for (short i=0; i<RAYS_REARRANGE_MULTIPLIER; i++)
  {
    const uint threadGlobalIndex = raySpatialData[threadLocalIndex() + i * threadGroupSize()].y;

    if (threadGlobalIndex != -1)
    {
      rays[threadGlobalIndex] = localRays[i];
    }
  }
}

Kernel void processShadowRays(
  Device colorType4*          colorOut,
  const Device RayStruct*     shadowRays,
  const Device HitStruct*     hits,
  constantKernelInput(uint,   rayCount),
  constantKernelInput(ushort, lastIteration)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint index = threadIndex();

  if (index >= rayCount)
    return;

  const HitStruct shadowHit = hits[index];

#ifdef RayStructColor
  if (shadowHit.primitiveIndex == -1)
  {
    const RayStruct shadowRay = shadowRays[index];
    colorType4 finalColor = colorOut[shadowRay.rayIndex];
    finalColor.xyz += shadowRay.color.xyz;
    finalColor.w = 1.f;

    colorOut[shadowRay.rayIndex] = finalColor;
  }
#endif
}

#endif
