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
      const float radius = extractPackedFloat(attributeBuffer, attributePackingInfo, index + primitivePackingInfo.elementOffset);

      outPrim.identity = primitiveIdentity;
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

      vertIndices = vertexOffset + select(constructUint3(0, 1, 2) + index * 3, vertIndices, selectInput3(attributePackingInfo.strideIn4Bytes == 0));
      finalVertexArray[vertIndices.x] = vert0;
      finalVertexArray[vertIndices.y] = vert1;
      finalVertexArray[vertIndices.z] = vert2;
    }
  }
}

/*
@kernel Shade ray intersection in a surface based on its material properties and visiblity info.
@param rays Ray buffer.
@param colorOut Color output buffer.
@param hits Hit info buffer.
@param rayCount Ray count.
*/
Kernel void shadeIntersection(
  Device RayStruct*             shadowRays,
  Device RayStruct*             rays,
  Device uint*                  colorOut,
  const Device HitStruct*       hits,
  constantKernelInput(uint,     rayCount),
  Const LightStruct*            lights,
  constantKernelInput(ushort,   lightOffset),
  constantKernelInput(ushort,   lightCount),
  const Device MaterialStruct*  materials
  KERNEL_GLOBAL_ARGUMENTS)
{
  uint index = threadIndex();

  if (index >= rayCount)
    return;

  const HitStruct hit = hits[index];
  RayStruct ray = rays[index];
  RayStruct shadowRay;

  for (ushort i=lightOffset; i<lightCount; i++)
  {
#ifdef RayStructColor
    shadowRay.color = constructColor4(0.f);
#endif
    if (hit.distance != INFINITY)
    {
      shadowRay.origin = ray.origin + ray.direction * hit.distance;

      float3 lightPosition, lightColor;
      sampleLight(&lightPosition, &lightColor, lights[i]);
      float3 direction = lightPosition - shadowRay.origin;

#ifdef HitStructNormal
      if (dot(direction, hit.normal) >= 0.f)
#endif
      {
        const float maxDistance = length(direction);
        direction /= maxDistance;
        shadowRay.maxDistance = maxDistance;
        shadowRay.direction = direction;
#ifdef RayStructColor
        const MaterialId materialId = hit.primitiveIdentity;
        shadowRay.color.xyz = constructColor3(lightColor.xyz) * ray.color.xyz * shadeMaterialAtIntersection(materials[materialId.identity], direction, hit).xyz;
#endif
      }
    }
    shadowRay.rayIndex  = ray.rayIndex;
    shadowRays[ray.rayIndex + rayCount * i] = shadowRay;
  }
}

Kernel void processShadowRays(
  Device uint*              colorOut,
  const Device RayStruct*   shadowRays,
  const Device HitStruct*   hits,
  constantKernelInput(uint, rayCount)
  KERNEL_GLOBAL_ARGUMENTS)
{
  uint index = threadIndex();

  if (index >= rayCount)
    return;

  const HitStruct shadowHit = hits[index];

#ifdef RayStructColor
  RayStruct shadowRay = shadowRays[index];
  if (shadowHit.primitiveIndex != -1)
  {
    shadowRay.color.xyz = 0.f;
  }
  colorType3 finalColor = 255.f * clamp(shadowRay.color.xyz, 0.f, 1.f);

  colorOut[shadowRay.rayIndex] = asUint(constructUchar4(constructUchar3(finalColor.xyz), 255));
#endif
}

#endif
