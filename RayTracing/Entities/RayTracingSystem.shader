#ifndef RAY_TRACING_SYSTEM_SHADER
#define RAY_TRACING_SYSTEM_SHADER

/*
@kernel Shade ray intersection in a surface based on its material properties and visiblity info.
@param rays Ray buffer.
@param colorOut Color output buffer.
@param hits Hit info buffer.
@param rayCount Ray count.
*/
Kernel void shadeIntersection(
  Device RayStruct*         rays,
  Device uint*              colorOut,
  const Device HitStruct*   hits,
  constantKernelInput(uint, rayCount)
  KERNEL_GLOBAL_ARGUMENTS)
{
  uint index = threadIndex();

  if (index >= rayCount)
    return;

  const HitStruct hit = hits[index];
  RayStruct ray = rays[index];

  if (hit.distance == INFINITY)
  {
    ray.color = constructFloat3(0.f, 0.f, 0.f);
  }
  else
  {
    ray.color = constructFloat3(1.f/hit.distance);
  }
  colorOut[ray.rayIndex] = asUint(constructUchar4(constructUchar3(255.f * clamp(ray.color, 0.f, 1.f)), 255));
}

#endif
