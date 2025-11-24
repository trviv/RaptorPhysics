/*
 * RaptorPhysics
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#ifndef ACCELERATION_DATA_STRUCT_TRAVERSE_SHADER
#define ACCELERATION_DATA_STRUCT_TRAVERSE_SHADER

#include "ComputeHeader.shader"
#include "ComputeShared.h"
#include "RayTracingStruct.h"
#include "HitStructs.h"

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
        setHitUV(hit->u, u);
        setHitUV(hit->v, v);
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
      setHitPrimitiveInternalIndex(hit->primitiveIndex, 0);
    }

    if (quadIndex.w != -1)
    {
      edge1 = edge2;
      edge2.position = vertexArray[quadIndex.w].position - vert0.position;

      if (triangleIntersection(hit, vert0, edge1.position, edge2.position, rayOrigin, rayDirection))
      {
        setHitPrimitiveInternalIndex(hit->primitiveIndex, 1);
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

inline void traversalSetHitNormal(
  const RayStruct               ray,
  Thread HitStruct*             hit,
  const Device PrimitiveStruct* vertexArray,
  const Device PrimitiveAttrib* attributeArray,
  const Device VertexAttrib*    vertexAttributeArray,
  Const RTSystemSettings*       systemSettings,
  const bool                    normalizeHit = true)
{
#if defined(HitStructIndex) && defined(HitStructIdentity) && defined(HitStructNormal)
  if (hit->primitiveIndex == -1) return;

  const uint primitiveIndex = getHitPrimitiveIndex(hit->primitiveIndex);
  DecodedPrimitiveInfo primInfo = defaultPrimitiveInfo();
  decodePrimitiveInfoFromSystemSettings(systemSettings, primitiveIndex, &primInfo);

  if (primInfo.primitiveType == PrimitiveSphere)
  {
    const float3 hitPoint = ray.origin + ray.direction * hit->distance;
    setHitNormal(hit->normal, hitPoint - vertexArray[primitiveIndex].position);
  }
  else if (primInfo.primitiveType == PrimitiveTriangle || primInfo.primitiveType == PrimitiveIndexedTriangle || primInfo.primitiveType == PrimitiveIndexedQuad)
  {
    PrimitiveAttrib attributes;
    float3 vert0, edge1, edge2;
    float3 normal0, normal1, normal2;

    if (primInfo.primitiveType == PrimitiveTriangle)
    {
      const uint triIndex = primInfo.vertexOffset + (primitiveIndex - primInfo.primitiveOffset)*3;
      attributes.triangleIndex = constructUint3(triIndex, triIndex+1, triIndex+2);
    }
    else if (primInfo.primitiveType == PrimitiveIndexedTriangle || primInfo.primitiveType == PrimitiveIndexedQuad)
    {
      attributes = attributeArray[primitiveIndex];
      if (primInfo.primitiveType == PrimitiveIndexedQuad && getHitPrimitiveInternalIndex(hit->primitiveIndex) == 1)
      {
        attributes.quadIndex.yz = attributes.quadIndex.zw;
      }
    }

    vert0 = vertexArray[attributes.triangleIndex.x].position;
    edge1 = vertexArray[attributes.triangleIndex.y].position;
    edge2 = vertexArray[attributes.triangleIndex.z].position;

    if (primInfo.primitiveType == PrimitiveIndexedTriangle || primInfo.primitiveType == PrimitiveIndexedQuad)
    {
      edge1 -= vert0;
      edge2 -= vert0;
    }

    if (isIdentityEntityFlat(hit->primitiveIdentity))
    {
      setHitNormal(hit->normal, cross(edge2, edge1));
    }
    else
    {
      const float3 tvec = ray.origin - vert0;
      const float3 pvec = cross(ray.direction, edge2);
      const float invDet= 1.f / dot(edge1, pvec);
      const float3 qvec = cross(tvec, edge1);

      float u = dot(tvec, pvec) * invDet;
      float v = dot(ray.direction, qvec) * invDet;

      setHitUV(u, hit->u);
      setHitUV(v, hit->v);

      normal0 = vertexAttributeArray[attributes.triangleIndex.x].normal;
      normal1 = vertexAttributeArray[attributes.triangleIndex.y].normal;
      normal2 = vertexAttributeArray[attributes.triangleIndex.z].normal;

      setHitNormal(hit->normal, (1 - u - v) * normal0 + u * normal1 + v * normal2);
    }
  }
  if (normalizeHit) setHitNormal(hit->normal, normalize(hit->normal));
#endif
}

inline void traversalStoreHit(
  Device HitStruct*       deviceHit,
  const Thread HitStruct  threadHit)
{
#ifdef IntersectionTypeClosest
  *deviceHit = threadHit;
#endif
#ifdef IntersectionTypeAny
  *deviceHit = threadHit;
  //setHitDistance(deviceHit->distance, threadHit.distance);
  //setHitPrimitiveIndex(deviceHit->primitiveIndex, threadHit.primitiveIndex);
  //setHitPrimitiveIdentity(deviceHit->primitiveIdentity, threadHit.primitiveIdentity);
#endif
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
  const Device VertexAttrib*    vertexAttributeArray,
  constantKernelInput(uint,     primitiveCount),
  Const RTSystemSettings*       systemSettings
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint index = threadIndex();

  if (index >= rayCount)
    return;

  const RayStruct ray = rays[index];
  const float3 invRayDirection = 1.f / ray.direction;
  const bool3 sign = selectInput3(ray.direction < 0.f);

  HitStruct hit = defaultHit(rays[index].maxDistance);
  DecodedPrimitiveInfo primInfo = defaultPrimitiveInfo();

  for (uint primIndex = 0; primIndex < primitiveCount; primIndex++)
  {
    if (rayXABIntersectTest(hit.distance, boundingBoxes[primIndex], ray.origin, invRayDirection, sign))
    {
      if (earliestIntersection(&hit, primIndex, ray.origin, ray.direction, invRayDirection, sign, vertexArray, attributeArray, systemSettings, &primInfo))
      {
#ifdef IntersectionTypeAny
        break;
#endif
      }
    }
  }

  traversalSetHitNormal(ray, &hit, vertexArray, attributeArray, vertexAttributeArray, systemSettings);
  traversalStoreHit(&hits[index], hit);
}

#endif
