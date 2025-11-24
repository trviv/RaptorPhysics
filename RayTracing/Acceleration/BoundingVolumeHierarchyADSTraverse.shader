/*
 * RaptorPhysics
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#ifndef BOUNDING_VOLUME_HIERARCHY_ADS_TRAVERSE_SHADER_H
#define BOUNDING_VOLUME_HIERARCHY_ADS_TRAVERSE_SHADER_H

#include "BoundingVolumeHierarchyStacklessTraverse.shader"
#include "BoundingVolumeHierarchyStackTraverse.shader"

Kernel void intersectRaysBVH(
  Device HitStruct*             hits,
  const Device RayStruct*       rays,
  constantKernelInput(uint,     rayCount),
  const Device PrimitiveStruct* vertexArray,
  const Device PrimitiveAttrib* attributeArray,
  const Device VertexAttrib*    vertexAttributeArray,
  const Device BVHNodeInfo*     treeInternalNodes,
  const Device uint*            leafParentNodeIndices,
  const Device uint*            nodeParentNodeIndices,
  const Device XAB*             treeLeafNodeBoundingBoxes,
  const Device XAB*             treeInternalNodeBoundingBoxes,
  Const RTSystemSettings*       systemSettings,
  constantKernelInput(uint,     primitiveCount),
  atomicKernelInput(uint,       rayIndexAtomicBuffer),
  sharedMemKernelInput(uint,    sharedLeafNodeIndex, 14)
  KERNEL_GLOBAL_ARGUMENTS
  KERNEL_THREAD_ARGUMENTS)
{
#if BVH_ADS_PERSISTENT_MULTIPLIER > 1
  volatile Shared uint nextRayArray[33];
  volatile Shared uint rayCountArray[33];

  const ushort simdLocalIndex = threadLocalIndex() & (ComputeSimdWidth - 1);
  const ushort simdGroupIndex = threadLocalIndex() >> ComputeSimdWidthExp;

  if (simdLocalIndex == 0)
  {
    rayCountArray[simdGroupIndex] = 0;
  }

  while (true)
  {
    // get rays from global to local pool
    if (rayCountArray[simdGroupIndex] == 0 && simdLocalIndex == 0)
    {
      nextRayArray[simdGroupIndex]  = atomicAdd((rayIndexAtomicBuffer+3), BVH_ADS_PERSISTENT_MULTIPLIER*ComputeSimdWidth);
      rayCountArray[simdGroupIndex] = BVH_ADS_PERSISTENT_MULTIPLIER*ComputeSimdWidth;
    }

    // get rays from local pool
    const uint index = nextRayArray[simdGroupIndex] + simdLocalIndex;
    if (index >= rayCount)
    {
      return;
    }

    if (simdLocalIndex == 0)
    {
      nextRayArray[simdGroupIndex]  += ComputeSimdWidth;
      rayCountArray[simdGroupIndex] -= ComputeSimdWidth;
    }
#else
  {
    const uint index = threadIndex();
    if (index >= rayCount) return;
#endif
    DecodedPrimitiveInfo primInfo = defaultPrimitiveInfo();

    const RayStruct ray = rays[index];
    const float3 invRayDirection = 1.f / ray.direction;
    const bool3 sign = selectInput3(ray.direction < 0.f);

    HitStruct hit = stacklessTraverseBinaryTree(rays[index].maxDistance,
      treeInternalNodes, leafParentNodeIndices, nodeParentNodeIndices, treeLeafNodeBoundingBoxes, treeInternalNodeBoundingBoxes,
      ray.origin, ray.direction, invRayDirection, sign,
      vertexArray, attributeArray, systemSettings,
      &primInfo, threadLocalIndex(), sharedLeafNodeIndex);

    traversalSetHitNormal(ray, &hit, vertexArray, attributeArray, vertexAttributeArray, systemSettings);
    traversalStoreHit(&hits[index], hit);
  }
}

#endif
