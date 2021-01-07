#ifndef RAY_TRACING_SYSTEM_SHADER
#define RAY_TRACING_SYSTEM_SHADER

Kernel void transformPrimitives(
  Device PrimitiveStruct*           vertexBuffer,
  constantKernelInput(PackingInfo,  primitivePackingInfo),
  constantKernelInput(uint,         maxVertexIndex),
  constantKernelInput(float4x4,     matrix)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint index = threadIndex();

  if (index > maxVertexIndex)
    return;

  PrimitiveStruct vertexPos = vertexBuffer[index + primitivePackingInfo.elementOffset];
  const IdentityInfo identity = vertexPos.identity;
  vertexPos.position = mulMatrixVec(matrix, constructFloat4(vertexPos.position, 1.f)).xyz;
  vertexPos.identity = identity;
  vertexBuffer[index + primitivePackingInfo.elementOffset] = vertexPos;
}

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
@param indexOffset Starting offset for storing vertex data.
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
  constantKernelInput(uint,         indexOffset)
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
      finalVertexArray[index + indexOffset] = outPrim;
      finalAttributeArray[index + indexOffset].radius = radius;
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

      vertIndices = indexOffset + constructUint3(0, 1, 2) + index * 3;
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
  const Device uint*            randomUints,
  constantKernelInput(uint,     rayCount),
  Const LightStruct*            lights,
  constantKernelInput(ushort,   lightOffset),
  constantKernelInput(ushort,   lightCount),
  const Device MaterialStruct*  materials,
  Const CameraStruct*           camera,
  constantKernelInput(uint,     iteration)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint index = threadIndex();

  if (index >= rayCount)
    return;

#if defined(RayStructColor) && !defined(HitStructIndex) && !defined(HitStructIdentity)
  HitStruct hit = hits[index];
  RayStruct ray = rays[index];
  RayStruct shadowRay;
  RayStruct childRay;

  const MaterialId materialId = hit.primitiveIdentity;
  MaterialStruct material;
  ushort materialType = -1;

  const uint pixelRandomValue = camera->frameIndex + randomUints[index];

  if (hit.primitiveIndex != -1)
  {
    shadowRay.origin = ray.origin + ray.direction * hit.distance;
    material = materials[removeIdentityFlags(materialId).identity];
    materialType = select(getMaterialType(material), (ushort)-1, isIdentityEntityNoShadow(materialId));
    if (isIdentityEntityTwoSided(materialId) && dot(ray.direction, hit.normal) >= 0.f)
    {
      hit.normal = -hit.normal;
    }
    colorType4 finalColor = colorOut[ray.rayIndex];
    finalColor.xyz += material.emissive.xyz * ray.color.xyz;
    finalColor.w = 1.f;
    colorOut[ray.rayIndex] = finalColor;
  }

  childRay = ray;
  childRay.origin = shadowRay.origin;

  if (materialType == MaterialTypeReflective)
  {
    childRay.direction = reflectVector(ray.direction, hit.normal);
    childRay.color *= material.specular;
  }
  else if (materialType == MaterialTypeTranslucent)
  {
    const float randomNumber = getRandomNumber(pixelRandomValue, 2+iteration*4);
    if (randomNumber < getFresnelCoefficient(dot(ray.direction, hit.normal), 1.f, getMaterialRefractiveIndex(material), getMaterialFresnelK(material)))
    {
      childRay.direction = reflectVector(ray.direction, hit.normal);
    }
    else
    {
      childRay.direction = refractVector(ray.direction, hit.normal, 1.f, getMaterialRefractiveIndex(material));
    }
    childRay.color *= material.specular;
  }
  else if (materialType == MaterialTypePlastic)
  {
    if (dot(ray.direction, hit.normal) <= 0.f)
    {
      const float2 random = constructFloat2(getRandomNumber(pixelRandomValue, 2+iteration*4+2), getRandomNumber(pixelRandomValue, 2+iteration*4+3));
      childRay.direction = alignHemisphereWithNormal(sampleCosineWeightedHemisphere(random), hit.normal);
      childRay.color *= material.diffuse;
    }
    else
    {
      materialType = -1;
    }
  }

  if (materialType != (ushort)-1)
  {
    childRay.maxDistance = select(INFINITY, 0.f, maxComp3(childRay.color) < MIN_TIME);
    childRay.rayIndex = ray.rayIndex;
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
      float3 lightColor, direction;
      float maxDistance;

      if (sampleLight(&lightColor, &direction, &maxDistance, lights + i, pixelRandomValue + i, iteration, shadowRay.origin)
        && dot(direction, hit.normal) >= MIN_TIME)
      {
        shadowRay.direction = direction;
        shadowRay.rayIndex  = ray.rayIndex;
        shadowRay.color.xyz = constructColor3(lightColor.xyz) * ray.color.xyz * shadeMaterialAtIntersection(material, direction, ray.direction, hit).xyz;
        shadowRay.maxDistance = select(maxDistance, 0.f, maxComp3(shadowRay.color) < MIN_TIME);
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

inline void bitonicSortSharedUshort2(
  Shared ushort2* localNodes,
  const short     maxDepth,
  const short     localThreadCount,
  const short     localIndex)
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
          const ushort2 node1 = localNodes[index];
          const ushort2 node2 = localNodes[swapIndex];

          if (node1.x > node2.x)
          {
            localNodes[index].xy     = node2;
            localNodes[swapIndex].xy = node1;
          }
        }
      }
      if (mergeSubSize > ComputeSimdWidth)
      {
        localMemBarrier();
      }
    }
  }
}

Kernel void reorderRays(
  Device RayStruct* raysOut,
  const Device RayStruct* raysIn,
  constantKernelInput(uint, rayCount),
  sharedMemKernelInput(ushort2, raySpatialData, 3)
  KERNEL_THREAD_ARGUMENTS
  KERNEL_THREADGROUP_ARGUMENTS)
{
  const uint groupOffset = threadGroupSize() * threadGroupIndex() * RAYS_REARRANGE_MULTIPLIER;

  for (short i=0; i<RAYS_REARRANGE_MULTIPLIER; i++)
  {
    ushort2 rayData;
    const uint threadGlobalIndex = threadLocalIndex() + threadGroupSize() * i + groupOffset;
    if (threadGlobalIndex < rayCount)
    {
      //const uint cellInternalSpatialIndex = encode32BitMortonCodeMath(constructInt3(511.f * (raysIn[threadGlobalIndex].direction + 1.f)));
      const ushort cellInternalSpatialIndex = encode16BitMortonCodeMath(constructShort3(15.f * (raysIn[threadGlobalIndex].direction + 1.f)));
      rayData = constructUshort2(cellInternalSpatialIndex, threadLocalIndex() + threadGroupSize() * i);
    }
    else
    {
      rayData = constructUshort2(-1);
    }
    raySpatialData[threadLocalIndex() + i * threadGroupSize()] = rayData;
  }

  const short tgSizePowOf2 = 32 - clz((int)threadGroupSize() * RAYS_REARRANGE_MULTIPLIER) - 1;
  bitonicSortSharedUshort2(raySpatialData, tgSizePowOf2, threadGroupSize(), threadLocalIndex());

  for (short i=0; i<RAYS_REARRANGE_MULTIPLIER; i++)
  {
    const ushort threadGlobalIndex = raySpatialData[threadLocalIndex() + i * threadGroupSize()].y;

    if (threadGlobalIndex != (ushort)-1)
    {
      raysOut[threadLocalIndex() + threadGroupSize() * i + groupOffset] = raysIn[groupOffset + threadGlobalIndex];
    }
  }
}

Kernel void processShadowRays(
  Device colorType4*        colorOut,
  const Device RayStruct*   shadowRays,
  const Device HitStruct*   hits,
  constantKernelInput(uint, rayCount)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint index = threadIndex();

  if (index >= rayCount)
    return;

  const HitStruct shadowHit = hits[index];

#ifdef RayStructColor
#ifdef HitStructIndex
  if (shadowHit.primitiveIndex == -1)
#endif
#ifdef HitStructIdentity
  if (isIdentityEntityNoShadow(shadowHit.primitiveIdentity))
#endif
  {
    const RayStruct shadowRay = shadowRays[index];
    colorType4 finalColor = colorOut[shadowRay.rayIndex];
    finalColor.xyz += shadowRay.color.xyz;
    finalColor.w = 1.f;

    colorOut[shadowRay.rayIndex] = finalColor;
  }
#endif
}

Kernel void accumulateColor(
  Device colorType4*        accumulatedColorOut,
  const Device colorType4*  colorOut,
  Const CameraStruct*       camera,
  constantKernelInput(uint, rayCount)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint index = threadIndex();

  if (index < rayCount)
  {
    if (camera->frameIndex > 0)
    {
      accumulatedColorOut[index] = (colorOut[index] + accumulatedColorOut[index] * (float)camera->frameIndex) / (float)(camera->frameIndex+1);
    }
    else
    {
      accumulatedColorOut[index] = colorOut[index];
    }
  }
}

#endif
