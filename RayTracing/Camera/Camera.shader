#ifndef CAMERA_SHADER
#define CAMERA_SHADER

inline RayStruct sampleCameraAtPixel(Const CameraStruct* camera, float pixelX, float pixelY)
{
  RayStruct worldRay;
  const float cW = camera->width;
  const float cH = camera->height;
  const float aspectRatio = cW / cH;

  /*const float4x4 mat = transpose(camera.viewMatrixInv);
  worldRay.origin = (mat * constructFloat4(0.f, 0.f, 0.f, 1.f)).xyz;
  const float x = (2.f * (pixelX + 0.5f) / cW - 1.f) * aspectRatio * camera.scale;
  const float y = (1.f - 2.f * (pixelY + 0.5f) / cH) * camera.scale;
  worldRay.direction = normalize((constructFloat4(x, y, -1.f, 0.f) * mat).xyz);*/

  const float4x4 mat = camera->viewMatrixInv;
  const float x = (2.f * (pixelX + 0.5f) / cW - 1.f) * camera->scale * aspectRatio;
  const float y = (1.f - 2.f * (pixelY + 0.5f) / cH) * camera->scale;

  worldRay.origin    = mulVecMatrix(constructFloat4(0.f, 0.f, 0.f, 1.f), mat).xyz;
  worldRay.direction = normalize(mulMatrixVec(mat, constructFloat4(x, y, -1.f, 0.f)).xyz);

  return worldRay;
}

/*
@kernel Emit primary rays from camera.
@param rays Ray buffer.
@param camera Camera to emit rays from.
*/
Kernel void emitPrimaryRaysLinear(
  Device RayStruct*   rays,
  Const CameraStruct* camera
  KERNEL_GLOBAL_ARGUMENTS)
{
  if (threadIndexN(0) >= camera->width || threadIndexN(1) >= camera->height)
    return;

  RayStruct ray = sampleCameraAtPixel(camera, threadIndexN(0), threadIndexN(1));

  // The camera emits primary rays
  ray.maxDistance = INFINITY;
  ray.rayIndex = threadIndexN(0) + camera->width * threadIndexN(1);
  ray.color = constructColor4(1.f);
  rays[ray.rayIndex] = ray;
}

inline uint decode32Bits2d(uint x)
{
  x &= 0x55555555;                 // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
  x = (x ^ (x >> 1)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
  x = (x ^ (x >> 2)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
  x = (x ^ (x >> 4)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
  x = (x ^ (x >> 8)) & 0x0000ffff; // x = ---- ---- ---- ---- fedc ba98 7654 3210

  return x;
}

inline short2 decode32BitMortonCode2d(const ushort c)
{
  const uint ret = decode32Bits2d(asUint(constructShort2(c, c >> 1)));
  return constructShort2(ret & 0xFF, ret >> 8);
}

/*
@kernel Emit primary rays from camera.
@param rays Ray buffer.
@param camera Camera to emit rays from.
*/
Kernel void emitPrimaryRaysZWalkLocal(
  Device RayStruct*   rays,
  Const CameraStruct* camera
  KERNEL_GLOBAL_ARGUMENTS
  KERNEL_THREAD_ARGUMENTS
  KERNEL_THREADGROUP_ARGUMENTS)
{
  if (threadIndexN(0) >= camera->width || threadIndexN(1) >= camera->height)
    return;

  const short2 xyOffset = constructShort2(threadIndexN(0) - threadLocalIndexN(0), threadIndexN(1) - threadLocalIndexN(1));
  const short2 xyIndex  = xyOffset + select(constructShort2(threadLocalIndexN(0), threadLocalIndexN(1)), decode32BitMortonCode2d(threadLocalIndex()),
    (camera->width - xyOffset.x) >= threadGroupSizeN(0) && (camera->height - xyOffset.y) >= threadGroupSizeN(1));

  const float2 random = constructFloat2(getRandomNumber(camera->frameIndex + xyIndex.x, 0), getRandomNumber(camera->frameIndex + xyIndex.y, 1));
  RayStruct ray   = sampleCameraAtPixel(camera, xyIndex.x + random.x, xyIndex.y + random.y);
  ray.maxDistance = INFINITY;
  ray.color       = constructColor4(1.f);
  ray.rayIndex    = xyIndex.x + camera->width * xyIndex.y;
  rays[threadIndexN(0) + camera->width * threadIndexN(1)] = ray;
}

/*
@kernel Emit primary rays from camera.
@param rays Ray buffer.
@param camera Camera to emit rays from.
*/
Kernel void emitPrimaryRaysZWalkLocalTG(
  Device RayStruct*   rays,
  Const CameraStruct* camera
  KERNEL_GLOBAL_ARGUMENTS
  KERNEL_THREAD_ARGUMENTS
  KERNEL_THREADGROUP_ARGUMENTS)
{
  short2 xyOffset = constructShort2(threadIndexN(0) - threadLocalIndexN(0), threadIndexN(1) - threadLocalIndexN(1));

  if ((threadGroupIndexN(0)&1) == 1 && (threadGroupIndexN(1)&1) == 0 && threadGroupIndexN(0) < (threadGroupCountN(0)-2) && threadGroupIndexN(1) < (threadGroupCountN(1)-2))
  {
    xyOffset += constructShort2(-threadGroupSizeN(0), threadGroupSizeN(1));
  }
  else if ((threadGroupIndexN(0)&1) == 0 && (threadGroupIndexN(1)&1) == 1 && threadGroupIndexN(0) < (threadGroupCountN(0)-1) && threadGroupIndexN(1) < (threadGroupCountN(1)-1))
  {
    xyOffset += constructShort2(threadGroupSizeN(0), -threadGroupSizeN(1));
  }

  const short2 index = xyOffset + constructShort2(threadLocalIndexN(0), threadLocalIndexN(1));

  if (index.x >= camera->width || index.y >= camera->height)
    return;

  const short2 xyIndex = xyOffset + select(constructShort2(threadLocalIndexN(0), threadLocalIndexN(1)), decode32BitMortonCode2d(threadLocalIndex()),
    (camera->width - xyOffset.x) >= threadGroupSizeN(0) && (camera->height - xyOffset.y) >= threadGroupSizeN(1));

  RayStruct ray   = sampleCameraAtPixel(camera, xyIndex.x, xyIndex.y);
  ray.maxDistance = INFINITY;
  ray.color       = constructColor4(1.f);
  ray.rayIndex    = xyIndex.x + camera->width * xyIndex.y;
  rays[threadIndexN(0) + camera->width * threadIndexN(1)] = ray;
}

#endif
