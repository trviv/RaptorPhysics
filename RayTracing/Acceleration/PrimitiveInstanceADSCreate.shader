#ifndef PRIMITIVE_INSTANCE_ACCELERATION_DATA_STRUCT_CREATE_SHADER
#define PRIMITIVE_INSTANCE_ACCELERATION_DATA_STRUCT_CREATE_SHADER

/*
@kernel Update per instance data based on instance transform.
@param primitiveInstanceNodes Primitive instance node data.
@param primitiveInstanceTransforms Primitive instance transforms and inverse transforms.
@param primitiveADSTreeInternalNodeBoundingBoxes Primitive acceleration data structure.
@param primitiveInstanceCount Total primitives in the buffer.
*/
Kernel void updatePrimitiveInstanceData(
  Device PrimitiveInstanceADSLeaf*    primitiveInstanceNodes,
  Device float4x4*                    primitiveInstanceTransforms,
  const Device PrimitiveADSResources* primitiveADSResources,
  constantKernelInput(uint,           primitiveInstanceCount)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint index = threadIndex();

  if (index >= primitiveInstanceCount) return;

  PrimitiveInstanceADSLeaf leafNode = primitiveInstanceNodes[index];

  const uint primitiveADSIndex = leafNode.primitiveADSIndex;
  const RayTracingEntityId primitiveInstance = leafNode.primitiveInstance;

  const XAB primitiveBoundingBox  = primitiveADSResources[primitiveADSIndex].systemSettings[0].systemBound;
  const float4x4 transform        = primitiveInstanceTransforms[index];

  float3 newMin = constructFloat3(INFINITY);
  float3 newMax = constructFloat3(-INFINITY);

  for (ushort i=0; i<2; i++)
  {
    for (ushort j=0; j<2; j++)
    {
      for (ushort k=0; k<2; k++)
      {
        float3 position = select(primitiveBoundingBox.min, primitiveBoundingBox.max, selectInput3(constructUshort3(i, j, k) > 0));
        position = mulMatrixVec(transform, constructFloat4(position, 1.f)).xyz;
        newMin = min(newMin, position);
        newMax = max(newMax, position);
      }
    }
  }

  leafNode.bounds.min = newMin;
  leafNode.bounds.max = newMax;
  leafNode.primitiveADSIndex = primitiveADSIndex;
  leafNode.primitiveInstance = primitiveInstance;

  primitiveInstanceNodes[index] = leafNode;

  float4x4 invTransform;

  const Thread float* matrix = (Thread float*)&transform;
  Thread float* inv = (Thread float*)&invTransform;

  inv[0] =    matrix[5]  * matrix[10] * matrix[15] -
              matrix[5]  * matrix[11] * matrix[14] -
              matrix[9]  * matrix[6]  * matrix[15] +
              matrix[9]  * matrix[7]  * matrix[14] +
              matrix[13] * matrix[6]  * matrix[11] -
              matrix[13] * matrix[7]  * matrix[10];

  inv[4] =    -matrix[4]  * matrix[10] * matrix[15] +
               matrix[4]  * matrix[11] * matrix[14] +
               matrix[8]  * matrix[6]  * matrix[15] -
               matrix[8]  * matrix[7]  * matrix[14] -
               matrix[12] * matrix[6]  * matrix[11] +
               matrix[12] * matrix[7]  * matrix[10];

  inv[8] =    matrix[4]  * matrix[9]  * matrix[15] -
              matrix[4]  * matrix[11] * matrix[13] -
              matrix[8]  * matrix[5]  * matrix[15] +
              matrix[8]  * matrix[7]  * matrix[13] +
              matrix[12] * matrix[5]  * matrix[11] -
              matrix[12] * matrix[7]  * matrix[9];

  inv[12] =   -matrix[4]  * matrix[9]  * matrix[14] +
               matrix[4]  * matrix[10] * matrix[13] +
               matrix[8]  * matrix[5]  * matrix[14] -
               matrix[8]  * matrix[6]  * matrix[13] -
               matrix[12] * matrix[5]  * matrix[10] +
               matrix[12] * matrix[6]  * matrix[9];

  inv[1] =    -matrix[1]  * matrix[10] * matrix[15] +
               matrix[1]  * matrix[11] * matrix[14] +
               matrix[9]  * matrix[2]  * matrix[15] -
               matrix[9]  * matrix[3]  * matrix[14] -
               matrix[13] * matrix[2]  * matrix[11] +
               matrix[13] * matrix[3]  * matrix[10];

  inv[5] =    matrix[0]  * matrix[10] * matrix[15] -
              matrix[0]  * matrix[11] * matrix[14] -
              matrix[8]  * matrix[2]  * matrix[15] +
              matrix[8]  * matrix[3]  * matrix[14] +
              matrix[12] * matrix[2]  * matrix[11] -
              matrix[12] * matrix[3]  * matrix[10];

  inv[9] =    -matrix[0]  * matrix[9]  * matrix[15] +
               matrix[0]  * matrix[11] * matrix[13] +
               matrix[8]  * matrix[1]  * matrix[15] -
               matrix[8]  * matrix[3]  * matrix[13] -
               matrix[12] * matrix[1]  * matrix[11] +
               matrix[12] * matrix[3]  * matrix[9];

  inv[13] =   matrix[0]  * matrix[9]  * matrix[14] -
              matrix[0]  * matrix[10] * matrix[13] -
              matrix[8]  * matrix[1]  * matrix[14] +
              matrix[8]  * matrix[2]  * matrix[13] +
              matrix[12] * matrix[1]  * matrix[10] -
              matrix[12] * matrix[2]  * matrix[9];

  inv[2] =    matrix[1]  * matrix[6] * matrix[15] -
              matrix[1]  * matrix[7] * matrix[14] -
              matrix[5]  * matrix[2] * matrix[15] +
              matrix[5]  * matrix[3] * matrix[14] +
              matrix[13] * matrix[2] * matrix[7] -
              matrix[13] * matrix[3] * matrix[6];

  inv[6] =    -matrix[0]  * matrix[6] * matrix[15] +
               matrix[0]  * matrix[7] * matrix[14] +
               matrix[4]  * matrix[2] * matrix[15] -
               matrix[4]  * matrix[3] * matrix[14] -
               matrix[12] * matrix[2] * matrix[7] +
               matrix[12] * matrix[3] * matrix[6];

  inv[10] =   matrix[0]  * matrix[5] * matrix[15] -
              matrix[0]  * matrix[7] * matrix[13] -
              matrix[4]  * matrix[1] * matrix[15] +
              matrix[4]  * matrix[3] * matrix[13] +
              matrix[12] * matrix[1] * matrix[7] -
              matrix[12] * matrix[3] * matrix[5];

  inv[14] =   -matrix[0]  * matrix[5] * matrix[14] +
               matrix[0]  * matrix[6] * matrix[13] +
               matrix[4]  * matrix[1] * matrix[14] -
               matrix[4]  * matrix[2] * matrix[13] -
               matrix[12] * matrix[1] * matrix[6] +
               matrix[12] * matrix[2] * matrix[5];

  inv[3] =    -matrix[1] * matrix[6] * matrix[11] +
               matrix[1] * matrix[7] * matrix[10] +
               matrix[5] * matrix[2] * matrix[11] -
               matrix[5] * matrix[3] * matrix[10] -
               matrix[9] * matrix[2] * matrix[7] +
               matrix[9] * matrix[3] * matrix[6];

  inv[7] =    matrix[0] * matrix[6] * matrix[11] -
              matrix[0] * matrix[7] * matrix[10] -
              matrix[4] * matrix[2] * matrix[11] +
              matrix[4] * matrix[3] * matrix[10] +
              matrix[8] * matrix[2] * matrix[7] -
              matrix[8] * matrix[3] * matrix[6];

  inv[11] =   -matrix[0] * matrix[5] * matrix[11] +
               matrix[0] * matrix[7] * matrix[9] +
               matrix[4] * matrix[1] * matrix[11] -
               matrix[4] * matrix[3] * matrix[9] -
               matrix[8] * matrix[1] * matrix[7] +
               matrix[8] * matrix[3] * matrix[5];

  inv[15] =   matrix[0] * matrix[5] * matrix[10] -
              matrix[0] * matrix[6] * matrix[9] -
              matrix[4] * matrix[1] * matrix[10] +
              matrix[4] * matrix[2] * matrix[9] +
              matrix[8] * matrix[1] * matrix[6] -
              matrix[8] * matrix[2] * matrix[5];

  const float invDet = 1.f / (matrix[0] * inv[0] + matrix[1] * inv[4] + matrix[2] * inv[8] + matrix[3] * inv[12]);
  primitiveInstanceTransforms[primitiveInstanceCount + index] = invTransform * invDet;
}

/*
@kernel Compute and store morton code for each primitive.
@param bvhLeafs Particle position and index data for bounding volume hierarchy.
@param vertexArray Primitive position array.
@param systemSettings Settings for the ray tracing system.
@param primitiveBatchSize Primitives processed per thread.
@param primitiveCount Total primitives in the buffer.
*/
Kernel void primitiveADSAssignMortonCode(
  Device BVHLeafInfo*                     bvhLeafs,
  const Device PrimitiveInstanceADSLeaf*  primitiveInstanceNodes,
  Const RTSystemSettings*                 systemSettings,
  constantKernelInput(uint,               primitiveCount)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint index = threadIndex();

  if (index >= primitiveCount) return;

  const float3 inverseMergedBoxSize = 1024.f / (systemSettings->systemBound.max - systemSettings->systemBound.min);
  const float3 mergedBoxCenter = (systemSettings->systemBound.min + systemSettings->systemBound.max) * 0.5f;

  const XAB bounds = primitiveInstanceNodes[index].bounds;
  const float3 center = (bounds.max + bounds.min) * 0.5f;

  // Quantize into integer coordinates
  // floor() is needed to prevent the center cell, at (0,0,0) from being twice the size
  const float3 positionRelativeToCenter = (center - mergedBoxCenter) * inverseMergedBoxSize;

  int3 quantizedPosition = convertInt3(select(floor(positionRelativeToCenter), positionRelativeToCenter, positionRelativeToCenter >= 0.0f));

  // Clamp coordinates into [-512, 511], then convert range from [-512, 511] to [0, 1023]
  quantizedPosition = max(constructInt3(-512), min(quantizedPosition, constructInt3(511))) + constructInt3(512);

  //Interleave bits(assign a morton code, also known as a z-curve)
  BVHLeafInfo bvhLeaf;
  bvhLeaf.mortonCode = encode32BitMortonCode(quantizedPosition);
  bvhLeaf.index = index;

  bvhLeafs[index] = bvhLeaf;
}

#endif
