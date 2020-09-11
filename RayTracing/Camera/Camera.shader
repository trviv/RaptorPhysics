#ifndef CAMERA_SHADER
#define CAMERA_SHADER

inline Ray sampleCameraAtPixel(constantKernelInput(CameraStruct, camera), float pixelX, float pixelY)
{
  Ray worldRay;
  worldRay.origin = camera.origin;
  worldRay.direction = normalize(camera.topLeft + camera.deltaX * pixelX + camera.deltaY * pixelY - camera.origin);
  return worldRay;
}

/*
@kernel Emit primary rays from camera.
@param rays Ray buffer.
@param camera Camera to emit rays from.
*/
Kernel void emitPrimaryRays(
  Device Ray*                       rays,
  constantKernelInput(CameraStruct, camera)
  KERNEL_GLOBAL_ARGUMENTS)
{
  // Since we aligned the thread count to the threadgroup size, the thread index may be out of bounds
  // of the render target size.
  if (threadIndexN(0) >= camera.width && threadIndexN(1) >= camera.height)
    return;

  // Pixel coordinates for this thread
  float2 pixel = constructFloat2(threadIndexN(0), threadIndexN(1));

  //pixel += r;

  // Map pixel coordinates to -1..1
  //float2 uv = (float2)pixel / float2(uniforms.width, uniforms.height);
  //uv = uv * 2.0f - 1.0f;

  Ray ray = sampleCameraAtPixel(camera, threadIndexN(0), threadIndexN(1));

  // The camera emits primary rays
  ray.type = RayTypePrimary;

  rays[threadIndexN(1) * camera.width + threadIndexN(0)] = ray;
}

#endif
