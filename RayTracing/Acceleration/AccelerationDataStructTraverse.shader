#ifndef ACCELERATION_DATA_STRUCT_TRAVERSE_SHADER
#define ACCELERATION_DATA_STRUCT_TRAVERSE_SHADER

inline bool earliestIntersection(
  Thread HitStruct* hit,
  const uint    primIndex,
  const float3  rayOrigin,
  const float3  rayDirection,
  const float3  invRayDirection,
  const bool3   sign,
  const Device PrimitiveStruct* vertexArray,
  const Device PrimitiveAttrib* attributeArray,
  Const RTSystemSettings* systemSettings,
  Thread DecodedPrimitiveInfo* prevPrimInfo)
{
  decodePrimitiveInfoFromSystemSettings(systemSettings, primIndex, prevPrimInfo);
  const DecodedPrimitiveInfo primInfo = *prevPrimInfo;

  if (primInfo.primitiveType == PrimitiveSphere)
  {
    const PrimitiveStruct sphere = vertexArray[primInfo.vertexOffset + primIndex - primInfo.primitiveOffset];
    const float radius = attributeArray[primIndex].radius;

    const float3 pvec = rayOrigin - sphere.position;
    const float b = dot(pvec, rayDirection);
    const float c = lengthSq(pvec) - sqr(radius);
    float d = b * b - c;

    if (d >= 0)
    {
      d = sqrt(d);
      float time = -(b + d);
      if (time > MIN_TIME && time < hit->distance)
      {
        setHitPrimitiveIndex(hit->primitiveIndex, primIndex);
        setHitPrimitiveIdentity(hit->primitiveIdentity, sphere.identity);
        setHitDistance(hit->distance, time);
        return true;
      }

      time = -(b - d);
      if (time > MIN_TIME && time < hit->distance)
      {
        setHitPrimitiveIndex(hit->primitiveIndex, primIndex);
        setHitPrimitiveIdentity(hit->primitiveIdentity, sphere.identity);
        setHitDistance(hit->distance, time);
        return true;
      }
    }
    return false;
  }
  else if (primInfo.primitiveType == PrimitiveTriangle || primInfo.primitiveType == PrimitiveIndexedTriangle)
  {
    PrimitiveStruct vert0;
    PrimitiveStruct edge1;
    PrimitiveStruct edge2;

    if (primInfo.primitiveType == PrimitiveTriangle)
    {
      const uint triIndex = primInfo.vertexOffset + (primIndex - primInfo.primitiveOffset)*3;

      vert0 = vertexArray[triIndex];
      edge1 = vertexArray[triIndex+1];
      edge2 = vertexArray[triIndex+2];
    }
    else
    {
      const uint3 triangleIndex = attributeArray[primIndex].triangleIndex;

      vert0 = vertexArray[triangleIndex.x];
      edge1 = vertexArray[triangleIndex.y];
      edge2 = vertexArray[triangleIndex.z];
      
      edge1.position -= vert0.position;
      edge2.position -= vert0.position;
    }

    const float3 tvec = rayOrigin - vert0.position;
    const float3 pvec = cross(rayDirection, edge2.position);
    const float invDet= 1.f/dot(edge1.position, pvec);
    const float u     = dot(tvec, pvec) * invDet;

    if (u >= 0.0f && u <= 1.0f)
    {
      const float3 qvec = cross(tvec, edge1.position);
      const float v = dot(rayDirection, qvec) * invDet;

      if (v >= 0.0f && (u + v) <= 1.0f)
      {
        const float time = dot(edge2.position, qvec) * invDet;
        if (time > MIN_TIME && time < hit->distance)
        {
          setHitPrimitiveIndex(hit->primitiveIndex, primIndex);
          setHitPrimitiveIdentity(hit->primitiveIdentity, vert0.identity);
          setHitDistance(hit->distance, time);
          return true;
        }
      }
    }
    return false;
  }

  return false;
}

/*
@kernel Intersect rays with primitives and fill hit info.
@param hits Hit info buffer.
@param rays Ray buffer.
@param rayCount Ray count.
@param boundingBoxes Bounding box for primitives.
@param vertexArray Buffer containing primitive positions.
@param attributeArray Buffer containing primitive attribute data.
@param primitiveCount Total primitives in the buffer.
@param systemSettings Settings for the ray tracing system.
*/
Kernel void intersectRays(
  Device HitStruct*             hits,
  const Device RayStruct*       rays,
  constantKernelInput(uint,     rayCount),
  const Device XAB*             boundingBoxes,
  const Device PrimitiveStruct* vertexArray,
  const Device PrimitiveAttrib* attributeArray,
  constantKernelInput(uint,     primitiveCount),
  Const RTSystemSettings*       systemSettings
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint index = threadIndex();

  if (index >= rayCount)
    return;

  const float3 rayOrigin = rays[index].origin;
  const float3 rayDirection = rays[index].direction;
  const float3 invRayDirection = 1.f / rayDirection;
  const bool3 sign = selectInput3(invRayDirection < 0.f);

  HitStruct hit;
  initializeHit(&hit);

  hit.distance = rays[index].maxDistance;

  DecodedPrimitiveInfo primInfo;
  primInfo.primitiveType = RTPrimitiveCount;

  for (uint primIndex = 0; primIndex < primitiveCount; primIndex++)
  {
    if (rayXABIntersectTest(hit.distance, boundingBoxes[primIndex], rayOrigin, invRayDirection, sign))
    {
      if (earliestIntersection(&hit, primIndex, rayOrigin, rayDirection, invRayDirection, sign, vertexArray, attributeArray, systemSettings, &primInfo))
      {
#ifdef IntersectionTypeAny
        break;
#endif
      }
    }
  }

#ifdef IntersectionTypeClosest
  hits[index] = hit;
#endif
#ifdef IntersectionTypeAny
  setHitPrimitiveIndex(hits[index].primitiveIndex, hit.primitiveIndex);
  setHitPrimitiveIdentity(hits[index].primitiveIdentity, hit.primitiveIdentity);
#endif
}

#endif
