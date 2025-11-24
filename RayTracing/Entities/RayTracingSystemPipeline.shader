/*
 * RaptorPhysics
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#ifndef RAY_TRACING_SYSTEM_PIPELINE_SHADER
#define RAY_TRACING_SYSTEM_PIPELINE_SHADER

#include "PrimitiveInstanceADSTraverse.shader"
#include "MaterialStruct.h"
#include "Light.shader"

inline void saveValidIndex(Device int* validRays, const bool isValid, const int index)
{
  // offset valid indices by 1 and set 0 for invalid
#ifdef USE_VALID_RAY_BUFFERS
  validRays[index] = isValid ? index + 1 : 0;
#endif
}

inline uint getValidRayIndex(const Device int* validRays, const uint index)
{
  // remove offset of 1
#ifdef USE_VALID_RAY_BUFFERS
  return validRays[index] - 1;
#else
  return index;
#endif
}

/*
@kernel Shade ray intersection in a surface based on its material properties and visiblity info.
@param colorOut Final color output.
@param shadowRays Shadow ray buffer.
@param rays Ray buffer.
@param hits Hit info buffer.
@param rayCount Ray count.
*/
inline void shadeIntersectionAndSave(
  Device int*                   validChildRays,
  Device int*                   validShadowRays,
  Device colorType4*            colorOut,
  Device RayStruct*             shadowRays,
  Device RayStruct*             rays,
  RayStruct                     ray,
  const HitStruct               hit,
  const Device uint*            randomUints,
  constantKernelInput(uint,     rayCount),
  Const LightStruct*            lights,
  constantKernelInput(ushort,   lightOffset),
  constantKernelInput(ushort,   lightCount),
  const Device MaterialStruct*  materials,
  Const CameraStruct*           camera,
  constantKernelInput(uint,     iteration),
  constantKernelInput(uint,     maxIterations),
  const uint                    index)
{
#if defined(RayStructColor) && defined(HitStructIndex) && defined(HitStructIdentity) && defined(HitStructNormal)
  RayStruct childRay;

  const MaterialId materialId = hit.primitiveIdentity;
  MaterialStruct material;
  MaterialTypes materialType = MaterialTypeMax;

  float3 hitNormal;
  if (hit.primitiveIndex != -1)
  {
    ray.origin = ray.origin + ray.direction * hit.distance;
    material = materials[removeIdentityFlags(materialId).identity];
    materialType = (MaterialTypes)select(getMaterialType(material), (ushort)MaterialTypeMax, isIdentityEntityNoShadow(materialId));
    colorType4 finalColor = colorOut[ray.rayIndex];
    finalColor.xyz += material.emissive.xyz * ray.color.xyz;
    setBVHHit(finalColor, hit.bvhHits * 0.001f);
    finalColor.w = 1.f;
    colorOut[ray.rayIndex] = finalColor;

    hitNormal = hit.normal;
    if (isIdentityEntityTwoSided(materialId) && dot(ray.direction, hitNormal) >= 0.f)
    {
      hitNormal = -hitNormal;
    }
  }

  childRay = ray;
  childRay.maxDistance = 0.f;

  uint pixelRandomValue;
  if (materialType != MaterialTypeMax)
  {
    pixelRandomValue = camera->frameIndex + randomUints[index];
  }

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
      materialType = MaterialTypeMax;
    }
  }

  if (materialType != MaterialTypeMax)
  {
    childRay.maxDistance = select(INFINITY, 0.f, maxComp3(childRay.color) < MIN_TIME);
    childRay.rayIndex = ray.rayIndex;
  }

  if (iteration < (maxIterations - 1))
  {
    const bool childRayValid = (childRay.maxDistance != 0.f);
    saveValidIndex(validChildRays, childRayValid, index);
#ifdef USE_VALID_RAY_BUFFERS
    if (childRayValid)
#endif
    rays[index] = childRay;
  }

  RayStruct shadowRay;
  shadowRay.origin = ray.origin;
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
    }

    const int shadowIndex = index + rayCount * i;
    const bool shadowRayValid = (shadowRay.maxDistance != 0.f);
    saveValidIndex(validShadowRays, shadowRayValid, shadowIndex);
#ifdef USE_VALID_RAY_BUFFERS
    if (shadowRayValid)
#endif
    shadowRays[shadowIndex] = shadowRay;
  }
#endif
}

inline void processShadowRay(
  Device colorType4*  colorOut,
  const RayStruct     shadowRay,
  const HitStruct     shadowHit)
{
#ifdef RayStructColor
#ifdef HitStructIndex
  if (shadowHit.primitiveIndex == -1)
#endif
#ifdef HitStructIdentity
  if (isIdentityEntityNoShadow(shadowHit.primitiveIdentity))
#endif
  {
    colorType4 finalColor = colorOut[shadowRay.rayIndex];
    finalColor.xyz += shadowRay.color.xyz;
    finalColor.w = 1.f;

    colorOut[shadowRay.rayIndex] = finalColor;
  }
#endif
}

Kernel void shadeIntersection(
  Device int*                   validChildRays,
  Device int*                   validShadowRays,
  Device colorType4*            colorOut,
  Device RayStruct*             shadowRays,
  Device RayStruct*             rays,
  const Device RayStruct*       inputRays,
  const Device HitStruct*       hits,
  const Device uint*            randomUints,
  constantKernelInput(uint,     rayCount),
  Const LightStruct*            lights,
  constantKernelInput(ushort,   lightOffset),
  constantKernelInput(ushort,   lightCount),
  const Device MaterialStruct*  materials,
  Const CameraStruct*           camera,
  constantKernelInput(uint,     iteration),
  constantKernelInput(uint,     maxIterations)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint index = threadIndex();

  if (index >= rayCount)
    return;

  shadeIntersectionAndSave(validChildRays, validShadowRays, colorOut, shadowRays, rays, inputRays[index], hits[index], randomUints, rayCount, lights, lightOffset, lightCount, materials, camera, iteration, maxIterations, index);
}

Kernel void intersectAndShade(
  Device int*                   validChildRays,
  Device int*                   validShadowRays,
  Device colorType4*            colorOut,
  Device RayStruct*             shadowRays,
  Device RayStruct*             rays,
  const Device RayStruct*       inputRays,
  const Device uint*            randomUints,
  constantKernelInput(uint,     rayCount),
  Const LightStruct*            lights,
  constantKernelInput(ushort,   lightOffset),
  constantKernelInput(ushort,   lightCount),
  const Device MaterialStruct*  materials,
  Const CameraStruct*           camera,
  constantKernelInput(uint,     iteration),
  constantKernelInput(uint,     maxIterations),
  Const PrimitiveInstanceADSResources*  primitiveInstanceADSResources
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint index = threadIndex();

  if (index >= rayCount)
    return;

  uint rayIndex = index;
#ifdef USE_VALID_RAY_BUFFERS
  if (iteration > 0)
  {
    rayIndex = getValidRayIndex(validChildRays, index);
  }
#endif

  const RayStruct ray = inputRays[rayIndex];
  const HitStruct hit = intersectRayPrimitiveInstanceADS(ray, primitiveInstanceADSResources);

  shadeIntersectionAndSave(validChildRays, validShadowRays, colorOut, shadowRays, rays, ray, hit, randomUints, rayCount, lights, lightOffset, lightCount, materials, camera, iteration, maxIterations, index);
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
  const Device int*         validShadowRays,
  constantKernelInput(uint, rayCount)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint index = threadIndex();

  if (index >= rayCount)
    return;

  const uint rayIndex = getValidRayIndex(validShadowRays, index);
  processShadowRay(colorOut, shadowRays[rayIndex], hits[rayIndex]);
}

Kernel void intersectAndProcessShadowRays(
  Device colorType4*        colorOut,
  const Device RayStruct*   shadowRays,
  const Device int*         validShadowRays,
  constantKernelInput(uint, rayCount),
  Const PrimitiveInstanceADSResources*  primitiveInstanceADSResources
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint index = threadIndex();

  if (index >= rayCount)
    return;

  const uint rayIndex = getValidRayIndex(validShadowRays, index);
  const RayStruct shadowRay = shadowRays[rayIndex];
  const HitStruct shadowHit = intersectRayPrimitiveInstanceADS(shadowRay, primitiveInstanceADSResources);

  processShadowRay(colorOut, shadowRay, shadowHit);
}

#endif
