#ifndef RAY_TRACING_SYSTEM_SHADER
#define RAY_TRACING_SYSTEM_SHADER

/*
@kernel Shade ray intersection in a surface based on its material properties and visiblity info.
@param rays Ray buffer.
@param colorOut Color output buffer.
@param hits Hit info buffer.
@param rayCount Ray count.
*/
//#autoArgumentBuffer
Kernel void shadeIntersection(
  Device RayStruct*             shadowRays,
  Device RayStruct*             rays,
  Device uint*                  colorOut,
  const Device HitStruct*       hits,
  constantKernelInput(uint,     rayCount),
  Const LightStruct*            lights,
  constantKernelInput(ushort,   lightOffset),
  constantKernelInput(ushort,   lightCount)
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
        shadowRay.direction = normalize(direction);
#ifdef RayStructColor
        shadowRay.color     = lightColor * ray.color;
#endif
      }
    }
    shadowRay.rayIndex  = ray.rayIndex;
    shadowRays[ray.rayIndex + rayCount * i] = shadowRay;
  }
}

//#autoArgumentBuffer
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

  const HitStruct hit = hits[index];

#ifdef RayStructColor
  RayStruct shadowRay = shadowRays[index];
  float3 finalColor = constructFloat3(0.f);

  if (hit.primitiveIndex == -1)
  {
    shadowRay.color = constructFloat3(1.f/hit.distance);
    finalColor = 255.f * clamp(shadowRay.color, 0.f, 1.f);
  }

  colorOut[shadowRay.rayIndex] = asUint(constructUchar4(constructUchar3(finalColor.x, finalColor.y, finalColor.z), 255));
#endif
}

#endif
