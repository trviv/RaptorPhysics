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
  Device float4*                      primitiveInstanceTransforms,
  const Device PrimitiveADSResources* primitiveADSResources,
  //const Device RayTracingPointerType* primitiveADSTreeInternalNodeBoundingBoxes,
  constantKernelInput(uint,           primitiveInstanceCount)
  KERNEL_THREAD_ARGUMENTS
  KERNEL_THREADGROUP_ARGUMENTS)
{
  const uint index = threadLocalIndex();

  if (index >= primitiveInstanceCount) return;

  PrimitiveInstanceADSLeaf leafNode           = primitiveInstanceNodes[index];
  const uint primitiveADSIndex                = leafNode.primitiveADSIndex;
  const RayTracingEntityId primitiveInstance  = leafNode.primitiveInstance;

  const uint instanceIndex        = getRayTracingInstanceId(primitiveInstance);
  //const XAB primitiveBoundingBox  = ((const Device XAB*)primitiveADSTreeInternalNodeBoundingBoxes[primitiveADSIndex].pointer)[0];
  const XAB primitiveBoundingBox  = primitiveADSResources[primitiveADSIndex].treeInternalNodeBoundingBoxes[0];
  const float4x4 transform        = unpackDeviceFloat3x4To4x4(&primitiveInstanceTransforms[index * 3]);

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

  float4x4 invTransform = transpose(transform) / determinant(transform);
  (((Thread float4*)&invTransform)[0]).w = (((Thread float4*)&invTransform)[3]).x;
  (((Thread float4*)&invTransform)[1]).w = (((Thread float4*)&invTransform)[3]).y;
  (((Thread float4*)&invTransform)[2]).w = (((Thread float4*)&invTransform)[3]).z;

  packFloat4x4ToDevice3x4(invTransform, &primitiveInstanceTransforms[(primitiveInstanceCount + index) * 3])
}

#endif
