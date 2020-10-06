#ifndef CAMERA_SHADER
#define CAMERA_SHADER

inline Ray sampleCameraAtPixel(constantKernelInput(CameraStruct, camera), float pixelX, float pixelY)
{
  Ray worldRay;
  const float cW = camera.width;
  const float cH = camera.height;
  const float aspectRatio = cW / cH;

  /*const float4x4 mat = transpose(camera.viewMatrixInv);
  worldRay.origin = (mat * constructFloat4(0.f, 0.f, 0.f, 1.f)).xyz;
  const float x = (2.f * (pixelX + 0.5f) / cW - 1.f) * aspectRatio * camera.scale;
  const float y = (1.f - 2.f * (pixelY + 0.5f) / cH) * camera.scale;
  worldRay.direction = normalize((constructFloat4(x, y, -1.f, 0.f) * mat).xyz);*/

  const float4x4 mat = camera.viewMatrixInv;
  const float x = (2.f * (pixelX + 0.5f) / cW - 1.f) * camera.scale * aspectRatio;
  const float y = (1.f - 2.f * (pixelY + 0.5f) / cH) * camera.scale;

  worldRay.origin    = mulVecMatrix(constructFloat4(0.f, 0.f, 0.f, 1.f), mat).xyz;
  worldRay.direction = normalize(mulMatrixVec(mat, constructFloat4(x, y, -1.f, 0.f)).xyz);

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
  if (threadIndexN(0) >= camera.width || threadIndexN(1) >= camera.height)
    return;

  Ray ray = sampleCameraAtPixel(camera, threadIndexN(0), threadIndexN(1));

  // The camera emits primary rays
  ray.type = RayTypePrimary;
  ray.rayIndex = threadIndexN(0) + camera.width * threadIndexN(1);
  ray.color = constructFloat3(1.f, 1.f, 1.f);
  rays[ray.rayIndex] = ray;
}

#endif
