#ifndef RAY_TRACING_SYSTEM_UTIL_SHADER
#define RAY_TRACING_SYSTEM_UTIL_SHADER

#include "RayTracingStruct.h"

Kernel void transformPrimitives(
  Device PrimitiveStruct*           vertexBuffer,
  constantKernelInput(PackingInfo,  primitivePackingInfo),
  constantKernelInput(uint,         maxVertexIndex),
  constantKernelInput(float4x4,     matrix)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint index = threadIndex();

  if (index > maxVertexIndex)
    return;

  PrimitiveStruct vertexPos = vertexBuffer[index + primitivePackingInfo.elementOffset];
  const IdentityInfo identity = vertexPos.identity;
  vertexPos.position = mulMatrixVec(matrix, constructFloat4(vertexPos.position, 1.f)).xyz;
  vertexPos.identity = identity;
  vertexBuffer[index + primitivePackingInfo.elementOffset] = vertexPos;
}

Kernel void accumulateColor(
  Device colorType4*        accumulatedColorOut,
  const Device colorType4*  colorOut,
  Const CameraStruct*       camera,
  constantKernelInput(uint, rayCount)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint index = threadIndex();

  if (index < rayCount)
  {
    if (camera->frameIndex > 0)
    {
      accumulatedColorOut[index] = (colorOut[index] + accumulatedColorOut[index] * (float)camera->frameIndex) / (float)(camera->frameIndex+1);
    }
    else
    {
      accumulatedColorOut[index] = colorOut[index];
    }
  }
}

Kernel void updateCameraKernel(
  Device CameraStruct* newCamera,
  Const CameraStruct* camera
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint index = threadIndex();

  if (index < 1)
  {
    CameraStruct currCamera = *camera;
    CameraStruct prevCamera = *newCamera;

    Thread float* currCameraMatPtr = (Thread float*)&currCamera.viewMatrixInv;
    Thread float* prevCameraMatPtr = (Thread float*)&prevCamera.viewMatrixInv;

    for (uint i=0; i<16; i++)
    {
      if (currCameraMatPtr[i] != prevCameraMatPtr[i])
      {
        currCamera.frameIndex = 0;
        *newCamera = currCamera;
        return;
      }
    }

    newCamera->frameIndex++;
  }
}

#endif
