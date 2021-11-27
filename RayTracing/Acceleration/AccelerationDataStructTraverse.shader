#ifndef ACCELERATION_DATA_STRUCT_TRAVERSE_SHADER
#define ACCELERATION_DATA_STRUCT_TRAVERSE_SHADER

bool triangleIntersection(
  Thread HitStruct* hit,
  const PrimitiveStruct vert0,
  const float3 edge1,
  const float3 edge2,
  const float3 rayOrigin,
  const float3 rayDirection)
{
  const float3 tvec = rayOrigin - vert0.position;
  const float3 pvec = cross(rayDirection, edge2);
  const float invDet= 1.f/dot(edge1, pvec);
  const float u     = dot(tvec, pvec) * invDet;

  if (u >= 0.f && u <= 1.f)
  {
    const float3 qvec = cross(tvec, edge1);
    const float v = dot(rayDirection, qvec) * invDet;

    if (v >= 0.f && (u + v) <= 1.f)
    {
      const float time = dot(edge2, qvec) * invDet;
      if (time > MIN_TIME && time < hit->distance)
      {
        setHitDistance(hit->distance, time);
        return true;
      }
    }
  }

  return false;
}

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
  else if (primInfo.primitiveType == PrimitiveTriangle || primInfo.primitiveType == PrimitiveIndexedTriangle || primInfo.primitiveType == PrimitiveIndexedQuad)
  {
    PrimitiveStruct vert0;
    PrimitiveStruct edge1;
    PrimitiveStruct edge2;

    uint4 quadIndex;

    if (primInfo.primitiveType == PrimitiveTriangle)
    {
      quadIndex = constructUint4(primInfo.vertexOffset + (primIndex - primInfo.primitiveOffset) * 3 + constructUint3(0, 1, 2), -1);
    }
    else if (primInfo.primitiveType == PrimitiveIndexedTriangle)
    {
      quadIndex = attributeArray[primIndex].quadIndex;
    }
    else if (primInfo.primitiveType == PrimitiveIndexedQuad)
    {
      quadIndex = attributeArray[primIndex].quadIndex;
    }

    vert0 = vertexArray[quadIndex.x];
    edge1 = vertexArray[quadIndex.y];
    edge2 = vertexArray[quadIndex.z];

    const IdentityInfo identity = vert0.identity;

    if (primInfo.primitiveType == PrimitiveIndexedTriangle || primInfo.primitiveType == PrimitiveIndexedQuad)
    {
      edge1.position -= vert0.position;
      edge2.position -= vert0.position;
    }

    bool isIntersecting = triangleIntersection(hit, vert0, edge1.position, edge2.position, rayOrigin, rayDirection);

    if (isIntersecting)
    {
      setHitPrimitiveInternalIndex(hit->primitiveInternalIndex, 0);
    }

    if (quadIndex.w != -1)
    {
      edge1 = edge2;
      edge2.position = vertexArray[quadIndex.w].position - vert0.position;

      if (triangleIntersection(hit, vert0, edge1.position, edge2.position, rayOrigin, rayDirection))
      {
        setHitPrimitiveInternalIndex(hit->primitiveInternalIndex, 1);
        isIntersecting = true;
      }
    }

    if (isIntersecting)
    {
      setHitPrimitiveIndex(hit->primitiveIndex, primIndex);
      setHitPrimitiveIdentity(hit->primitiveIdentity, identity);
    }

    return isIntersecting;
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
