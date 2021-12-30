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
@param finalVertexAttributeArray Buffer containing vertex attribute data.
@param primitiveBuffer Buffer containing primitive positions.
@param primitivePackingInfo Packing information for primitive structure.
@param attributeBuffer Buffer containing attribute inside a structure.
@param attributePackingInfo Packing information for attribute in primitive structure.
@param vertexAttribBuffer Buffer containing vertex attribute inside a structure.
@param vertexAttribPackingInfo Packing information for vertex structure.
@param primitiveBatchSize Primitives processed per thread.
@param primitiveCount Total primitives in the buffer.
@param primitiveType Primitive type for the dispatch.
@param vertexOffset Starting offset for storing vertex data.
*/
Kernel void collectPrimitives(
  Device PrimitiveStruct*           finalVertexArray,
  Device PrimitiveAttrib*           finalAttributeArray,
  Device VertexAttrib*              finalVertexAttributeArray,
  const Device PrimitiveStruct*     primitiveBuffer,
  constantKernelInput(PackingInfo,  primitivePackingInfo),
  const Device float*               attributeBuffer,
  constantKernelInput(PackingInfo,  attributePackingInfo),
  const Device float*               vertexAttribBuffer,
  constantKernelInput(PackingInfo,  vertexAttribPackingInfo),
  constantKernelInput(IdentityInfo, primitiveIdentity),
  constantKernelInput(uint,         primitiveBatchSize),
  constantKernelInput(uint,         primitiveCount),
  constantKernelInput(uint,         primitiveType),
  constantKernelInput(uint,         primitiveOffset),
  constantKernelInput(uint,         vertexOffset),
  constantKernelInput(float4x4,     matrix)
  KERNEL_THREAD_ARGUMENTS
  KERNEL_THREADGROUP_ARGUMENTS)
{
  const Device PrimitiveAttrib* primitiveAttribPtr  = (const Device PrimitiveAttrib*)extractPackedPointer(attributeBuffer, attributePackingInfo);
  const Device VertexAttrib* vertexAttribPtr = (const Device VertexAttrib*)extractPackedPointer(vertexAttribBuffer, vertexAttribPackingInfo);

  uint index = threadLocalIndex() + primitiveBatchSize * threadGroupIndex() * threadGroupSize();
  for (short b = 0; index < primitiveCount && b < primitiveBatchSize; index += threadGroupSize(), b++)
  {
    if (primitiveType == PrimitiveSphere)
    {
      PrimitiveStruct outPrim  = primitiveBuffer[index + primitivePackingInfo.elementOffset];

      outPrim.position = mulMatrixVec(matrix, constructFloat4(outPrim.position, 1.f)).xyz;
      outPrim.identity = primitiveIdentity;
      finalVertexArray[index + vertexOffset] = outPrim;
      finalAttributeArray[index + vertexOffset].radius = primitiveAttribPtr[index].radius;
    }
    else if (primitiveType == PrimitiveIndexedTriangle || primitiveType == PrimitiveTriangle)
    {
      uint3 vertIndices;
      if (primitiveType == PrimitiveIndexedTriangle)
      {
        vertIndices = primitiveAttribPtr[index].triangleIndex;
      }
      else
      {
        vertIndices = constructUint3(0, 1, 2) + index * 3;
        // for indexed array a non zero stride is assumed
        if (attributePackingInfo.strideIn4Bytes > 0)
        {
          vertIndices = primitiveAttribPtr[index].triangleIndex;
        }
      }

      // get vertex zero and vertex position
      PrimitiveStruct vert0 = primitiveBuffer[vertIndices.x];
      PrimitiveStruct vert1 = primitiveBuffer[vertIndices.y];
      PrimitiveStruct vert2 = primitiveBuffer[vertIndices.z];

      vert0.position = mulMatrixVec(matrix, constructFloat4(vert0.position, 1.f)).xyz;
      vert1.position = mulMatrixVec(matrix, constructFloat4(vert1.position, 1.f)).xyz;
      vert2.position = mulMatrixVec(matrix, constructFloat4(vert2.position, 1.f)).xyz;

      const VertexAttrib vertAttrib0 = vertexAttribPtr[vertIndices.x];
      const VertexAttrib vertAttrib1 = vertexAttribPtr[vertIndices.y];
      const VertexAttrib vertAttrib2 = vertexAttribPtr[vertIndices.z];

      if (primitiveType == PrimitiveIndexedTriangle)
      {
        vertIndices += vertexOffset;
      }
      else
      {
        vert1.position = vert1.position - vert0.position;
        vert2.position = vert2.position - vert0.position;
        vertIndices = vertexOffset + constructUint3(0, 1, 2) + index * 3;
      }

      vert0.identity = primitiveIdentity;
      vert1.identity = primitiveIdentity;
      vert2.identity = primitiveIdentity;

      finalVertexArray[vertIndices.x] = vert0;
      finalVertexArray[vertIndices.y] = vert1;
      finalVertexArray[vertIndices.z] = vert2;

      finalVertexAttributeArray[vertIndices.x] = vertAttrib0;
      finalVertexAttributeArray[vertIndices.y] = vertAttrib1;
      finalVertexAttributeArray[vertIndices.z] = vertAttrib2;

      finalAttributeArray[index + primitiveOffset].quadIndex = constructUint4(vertIndices, -1);
    }
    else if (primitiveType == PrimitiveIndexedQuad)
    {
      uint4 vertIndices;
      {
        vertIndices = primitiveAttribPtr[index].quadIndex;
      }

      // get vertex zero and vertex position
      PrimitiveStruct vert0 = primitiveBuffer[vertIndices.x];
      PrimitiveStruct vert1 = primitiveBuffer[vertIndices.y];
      PrimitiveStruct vert2 = primitiveBuffer[vertIndices.z];
      PrimitiveStruct vert3;

      vert0.position = mulMatrixVec(matrix, constructFloat4(vert0.position, 1.f)).xyz;
      vert1.position = mulMatrixVec(matrix, constructFloat4(vert1.position, 1.f)).xyz;
      vert2.position = mulMatrixVec(matrix, constructFloat4(vert2.position, 1.f)).xyz;

      const VertexAttrib vertAttrib0 = vertexAttribPtr[vertIndices.x];
      const VertexAttrib vertAttrib1 = vertexAttribPtr[vertIndices.y];
      const VertexAttrib vertAttrib2 = vertexAttribPtr[vertIndices.z];
      VertexAttrib vertAttrib3;

      if (vertIndices.w != -1)
      {
        vert3 = primitiveBuffer[vertIndices.w];
        vert3.position = mulMatrixVec(matrix, constructFloat4(vert3.position, 1.f)).xyz;
        vertAttrib3 = vertexAttribPtr[vertIndices.w];
        vertIndices.w += vertexOffset;
        vert3.identity = primitiveIdentity;
      }

      if (primitiveType == PrimitiveIndexedQuad)
      {
        vertIndices.xyz += vertexOffset;
      }

      vert0.identity = primitiveIdentity;
      vert1.identity = primitiveIdentity;
      vert2.identity = primitiveIdentity;

      finalVertexArray[vertIndices.x] = vert0;
      finalVertexArray[vertIndices.y] = vert1;
      finalVertexArray[vertIndices.z] = vert2;

      if (vertIndices.w != -1)
      {
        finalVertexArray[vertIndices.w] = vert3;
        finalVertexAttributeArray[vertIndices.w] = vertAttrib3;
      }

      finalVertexAttributeArray[vertIndices.x] = vertAttrib0;
      finalVertexAttributeArray[vertIndices.y] = vertAttrib1;
      finalVertexAttributeArray[vertIndices.z] = vertAttrib2;

      finalAttributeArray[index + primitiveOffset].quadIndex = vertIndices;
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
  const Device PrimitiveStruct* vertexArray,
  const Device PrimitiveAttrib* attributeArray,
  const Device VertexAttrib*    vertexAttributeArray,
  constantKernelInput(uint,     rayCount),
  Const LightStruct*            lights,
  constantKernelInput(ushort,   lightOffset),
  constantKernelInput(ushort,   lightCount),
  const Device MaterialStruct*  materials,
  Const CameraStruct*           camera,
  constantKernelInput(uint,     iteration),
  Const RTSystemSettings*       systemSettings
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

  float3 hitNormal;
  if (hit.primitiveIndex != -1)
  {
    shadowRay.origin = ray.origin + ray.direction * hit.distance;

    DecodedPrimitiveInfo primInfo;
    primInfo.primitiveType = RTPrimitiveCount;
    decodePrimitiveInfoFromSystemSettings(systemSettings, hit.primitiveIndex, &primInfo);

    if (primInfo.primitiveType == PrimitiveSphere)
    {
      setHitNormal(hitNormal, shadowRay.origin - vertexArray[hit.primitiveIndex].position);
    }
    else if (primInfo.primitiveType == PrimitiveTriangle || primInfo.primitiveType == PrimitiveIndexedTriangle || primInfo.primitiveType == PrimitiveIndexedQuad)
    {
      PrimitiveAttrib attributes;
      float3 vert0, edge1, edge2;
      float3 normal0, normal1, normal2;

      if (primInfo.primitiveType == PrimitiveTriangle)
      {
        const uint triIndex = primInfo.vertexOffset + (hit.primitiveIndex - primInfo.primitiveOffset)*3;
        attributes.triangleIndex = constructUint3(triIndex, triIndex+1, triIndex+2);
      }
      else if (primInfo.primitiveType == PrimitiveIndexedTriangle)
      {
        attributes = attributeArray[hit.primitiveIndex];
      }
      else if (primInfo.primitiveType == PrimitiveIndexedQuad)
      {
        attributes = attributeArray[hit.primitiveIndex];
#if defined(HitStructIndexIdentity)
        if (hit.primitiveInternalIndex == 1)
        {
          attributes.quadIndex.y = attributes.quadIndex.x;
          attributes.quadIndex.x = attributes.quadIndex.w;
        }
#endif
      }

      vert0 = vertexArray[attributes.triangleIndex.x].position;
      edge1 = vertexArray[attributes.triangleIndex.y].position;
      edge2 = vertexArray[attributes.triangleIndex.z].position;

      if (primInfo.primitiveType == PrimitiveIndexedTriangle || primInfo.primitiveType == PrimitiveIndexedQuad)
      {
        edge1 -= vert0;
        edge2 -= vert0;
      }

      if (isIdentityEntityFlat(materialId))
      {
        setHitNormal(hitNormal, cross(edge2, edge1));
      }
      else
      {
        const float3 tvec = ray.origin - vert0;
        const float3 pvec = cross(ray.direction, edge2);
        const float invDet= 1.f / dot(edge1, pvec);
        const float3 qvec = cross(tvec, edge1);

        const float u = dot(tvec, pvec) * invDet;
        const float v = dot(ray.direction, qvec) * invDet;

        normal0 = vertexAttributeArray[attributes.triangleIndex.x].normal;
        normal1 = vertexAttributeArray[attributes.triangleIndex.y].normal;
        normal2 = vertexAttributeArray[attributes.triangleIndex.z].normal;

        setHitNormal(hitNormal, (1 - u - v) * normal0 + u * normal1 + v * normal2);
      }
    }
    hitNormal = normalize(hitNormal);

    material = materials[removeIdentityFlags(materialId).identity];
    materialType = select(getMaterialType(material), (ushort)-1, isIdentityEntityNoShadow(materialId));
    if (isIdentityEntityTwoSided(materialId) && dot(ray.direction, hitNormal) >= 0.f)
    {
      hitNormal = -hitNormal;
    }
    colorType4 finalColor = colorOut[ray.rayIndex];
    finalColor.xyz += material.emissive.xyz * ray.color.xyz;
#if defined(HitStructBVHHits)
    finalColor = hit.bvhHits * 0.001f;
#endif
    finalColor.w = 1.f;
    colorOut[ray.rayIndex] = finalColor;
  }

  childRay = ray;
  childRay.origin = shadowRay.origin;

  if (materialType == MaterialTypeReflective)
  {
    childRay.direction = reflectVector(ray.direction, hitNormal);
    childRay.color *= material.specular;
  }
  else if (materialType == MaterialTypeTranslucent)
  {
    const float randomNumber = getRandomNumber(pixelRandomValue, 2+iteration*4);
    if (randomNumber < getFresnelCoefficient(dot(ray.direction, hitNormal), 1.f, getMaterialRefractiveIndex(material), getMaterialFresnelK(material)))
    {
      childRay.direction = reflectVector(ray.direction, hitNormal);
    }
    else
    {
      childRay.direction = refractVector(ray.direction, hitNormal, 1.f, getMaterialRefractiveIndex(material));
    }
    childRay.color *= material.specular;
  }
  else if (materialType == MaterialTypePlastic)
  {
    if (dot(ray.direction, hitNormal) <= 0.f)
    {
      const float2 random = constructFloat2(getRandomNumber(pixelRandomValue, 2+iteration*4+2), getRandomNumber(pixelRandomValue, 2+iteration*4+3));
      childRay.direction = alignHemisphereWithNormal(sampleCosineWeightedHemisphere(random), hitNormal);
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
        && dot(direction, hitNormal) >= MIN_TIME)
      {
        shadowRay.direction = direction;
        shadowRay.rayIndex  = ray.rayIndex;
        shadowRay.color.xyz = constructColor3(lightColor.xyz) * ray.color.xyz * shadeMaterialAtIntersection(material, direction, ray.direction, hit, hitNormal).xyz;
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

Kernel void updateCameraKernel(
  Device CameraStruct* newCamera,
  Const CameraStruct* camera
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint index = threadIndex();

  if (index < 1)
  {
    CameraStruct currCamera = *camera;
    CameraStruct prevCamera = *newCamera;

    Thread float* currCameraMatPtr = (Thread float*)&currCamera.viewMatrixInv;
    Thread float* prevCameraMatPtr = (Thread float*)&prevCamera.viewMatrixInv;

    for (uint i=0; i<16; i++)
    {
      if (currCameraMatPtr[i] != prevCameraMatPtr[i])
      {
        currCamera.frameIndex = 0;
        *newCamera = currCamera;
        return;
      }
    }

    newCamera->frameIndex++;
  }
}

#endif
