#ifndef ACCELERATION_DATA_STRUCT_TRAVERSE_SHADER
#define ACCELERATION_DATA_STRUCT_TRAVERSE_SHADER

inline void earliestIntersection(
  Thread HitStruct* hit,
  const uint    primIndex,
  const float3  rayOrigin,
  const float3  rayDirection,
  const float3  invRayDirection,
  const bool3   sign,
  const Device PrimitiveStruct* vertexArray,
  const Device PrimitiveAttrib* attributeArray,
  Const RTSystemSettings* systemSettings)
{
  const DecodedPrimitiveInfo primInfo = decodePrimitiveInfoFromSystemSettings(systemSettings, primIndex);

  if (primInfo.primType == PrimitiveSphere)
  {
    const PrimitiveStruct sphere = vertexArray[primInfo.vertexOffset + primIndex - primInfo.indexOffset];
    const float radius = attributeArray[primIndex].radius;

    const float3 pvec = rayOrigin - sphere.position;
    const float b = dot(pvec, rayDirection);
    const float c = lengthSq(pvec) - sqr(radius);
    float d = b * b - c;

    if (d >= 0)
    {
      d = sqrt(b * b - c);
      float time = -(b + d);
      if (time > MIN_TIME && time < hit->distance)
      {
        hit->distance = time;
        hit->primitiveIndex = primIndex;
      }

      time = -(b - d);
      if (time > MIN_TIME && time < hit->distance)
      {
        hit->distance = time;
        hit->primitiveIndex = primIndex;
      }
    }
  }

  if (primInfo.primType == PrimitiveTriangle)
  {
    const uint triIndex = primInfo.vertexOffset + (primIndex - primInfo.indexOffset)*3;

    const PrimitiveStruct vert0 = vertexArray[triIndex];
    const PrimitiveStruct edge1 = vertexArray[triIndex+1];
    const PrimitiveStruct edge2 = vertexArray[triIndex+2];

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
        hit->distance = dot(edge2.position, qvec) * invDet;
        hit->primitiveIndex = primIndex;
      }
    }
  }
}

inline bool anyIntersection(
  const Thread HitStruct* hit,
  const uint    primIndex,
  const float3  rayOrigin,
  const float3  rayDirection,
  const float3  invRayDirection,
  const bool3   sign,
  const Device PrimitiveStruct* vertexArray,
  const Device PrimitiveAttrib* attributeArray,
  Const RTSystemSettings* systemSettings)
{
  const DecodedPrimitiveInfo primInfo = decodePrimitiveInfoFromSystemSettings(systemSettings, primIndex);

  if (primInfo.primType == PrimitiveSphere)
  {
    const PrimitiveStruct sphere = vertexArray[primInfo.vertexOffset + primIndex - primInfo.indexOffset];
    const float radius = attributeArray[primIndex].radius;

    const float3 pvec = rayOrigin - sphere.position;
    const float b = dot(pvec, rayDirection);
    const float c = lengthSq(pvec) - sqr(radius);
    float d = b * b - c;

    if (d >= 0)
    {
      d = sqrt(b * b - c);
      float time = -(b + d);
      if (time > MIN_TIME && time < hit->distance)
      {
        return true;
      }

      time = -(b - d);
      if (time > MIN_TIME && time < hit->distance)
      {
        return true;
      }
    }
  }

  if (primInfo.primType == PrimitiveTriangle)
  {
    const uint triIndex = primInfo.vertexOffset + (primIndex - primInfo.indexOffset)*3;

    const PrimitiveStruct vert0 = vertexArray[triIndex];
    const PrimitiveStruct edge1 = vertexArray[triIndex+1];
    const PrimitiveStruct edge2 = vertexArray[triIndex+2];

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
        return true;
      }
    }
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
  Const RTSystemSettings*       systemSettings,
  constantKernelInput(ushort,   initHit)
  KERNEL_GLOBAL_ARGUMENTS)
{
  uint index = threadIndex();

  if (index >= rayCount)
    return;

  const float3 rayOrigin = rays[index].origin;
  const float3 rayDirection = rays[index].direction;
  const float3 invRayDirection = 1.f / rayDirection;
  const bool3 sign = selectInput3(invRayDirection < 0.f);

  HitStruct hit;

  if (initHit)
  {
    initializeHit(&hit);
  }
  else
  {
    hit = hits[index];
    hit.primitiveIndex = -1;
  }

  float currentTime = hit.distance;

  for (uint primIndex = 0; primIndex < primitiveCount; primIndex++)
  {
    if (rayXABIntersectEarliest(&currentTime, boundingBoxes[primIndex], rayOrigin, invRayDirection, sign))
    {
#ifdef IntersectionTypeClosest
      earliestIntersection(&hit, primIndex, rayOrigin, rayDirection, invRayDirection, sign, vertexArray, attributeArray, systemSettings);
      currentTime = hit.distance;
#endif
#ifdef IntersectionTypeAny
      if (anyIntersection(&hit, primIndex, rayOrigin, rayDirection, invRayDirection, sign, vertexArray, attributeArray, systemSettings))
      {
        hit.primitiveIndex = 0;
        break;
      }
#endif
    }
  }

#ifdef IntersectionTypeClosest
  hits[index] = hit;
#endif
#ifdef IntersectionTypeAny
  hits[index].primitiveIndex = hit.primitiveIndex;
#endif
}

#endif
