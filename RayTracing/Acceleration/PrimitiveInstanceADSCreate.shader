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
  constantKernelInput(uint,           primitiveInstanceCount)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint index = threadIndex();

  if (index >= primitiveInstanceCount) return;

  PrimitiveInstanceADSLeaf leafNode = primitiveInstanceNodes[index];

  const uint primitiveADSIndex = leafNode.primitiveADSIndex;
  const RayTracingEntityId primitiveInstance = leafNode.primitiveInstance;

  const uint instanceIndex        = getRayTracingInstanceId(primitiveInstance);
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

/*
@kernel Create single array composed of all the primitives.
@param finalVertexArray Buffer containing all positions.
@param finalAttributeArray Buffer containing primitive attribute data.
@param finalVertexAttributeArray Buffer containing vertex attribute data.
@param primitiveBuffer Buffer containing primitive positions.
@param primitivePackingInfo Packing information for primitive structure.
@param attributeBuffer Buffer containing attribute inside a structure.
@param attributePackingInfo Packing information for attribute in primitive structure.
@param vertexAttribBuffer Buffer containing vertex attribute inside a structure.
@param vertexAttribPackingInfo Packing information for vertex structure.
@param primitiveBatchSize Primitives processed per thread.
@param primitiveCount Total primitives in the buffer.
@param primitiveType Primitive type for the dispatch.
@param vertexOffset Starting offset for storing vertex data.
*/
Kernel void collectPrimitives(
  Device PrimitiveStruct*           finalVertexArray,
  Device PrimitiveAttrib*           finalAttributeArray,
  Device VertexAttrib*              finalVertexAttributeArray,
  const Device PrimitiveStruct*     primitiveBuffer,
  constantKernelInput(PackingInfo,  primitivePackingInfo),
  const Device float*               attributeBuffer,
  constantKernelInput(PackingInfo,  attributePackingInfo),
  const Device float*               vertexAttribBuffer,
  constantKernelInput(PackingInfo,  vertexAttribPackingInfo),
  constantKernelInput(IdentityInfo, primitiveIdentity),
  constantKernelInput(uint,         primitiveBatchSize),
  constantKernelInput(uint,         primitiveCount),
  constantKernelInput(uint,         primitiveType),
  constantKernelInput(uint,         primitiveOffset),
  constantKernelInput(uint,         vertexOffset),
  constantKernelInput(float4x4,     matrix)
  KERNEL_THREAD_ARGUMENTS
  KERNEL_THREADGROUP_ARGUMENTS)
{
  const Device PrimitiveAttrib* primitiveAttribPtr  = (const Device PrimitiveAttrib*)extractPackedPointer(attributeBuffer, attributePackingInfo);
  const Device VertexAttrib* vertexAttribPtr = (const Device VertexAttrib*)extractPackedPointer(vertexAttribBuffer, vertexAttribPackingInfo);

  uint index = threadLocalIndex() + primitiveBatchSize * threadGroupIndex() * threadGroupSize();
  for (short b = 0; index < primitiveCount && b < primitiveBatchSize; index += threadGroupSize(), b++)
  {
    if (primitiveType == PrimitiveSphere)
    {
      PrimitiveStruct outPrim  = primitiveBuffer[index + primitivePackingInfo.elementOffset];

      outPrim.position = mulMatrixVec(matrix, constructFloat4(outPrim.position, 1.f)).xyz;
      outPrim.identity = primitiveIdentity;
      finalVertexArray[index + vertexOffset] = outPrim;
      finalAttributeArray[index + vertexOffset].radius = primitiveAttribPtr[index].radius;
    }
    else if (primitiveType == PrimitiveIndexedTriangle || primitiveType == PrimitiveTriangle)
    {
      uint3 vertIndices;
      if (primitiveType == PrimitiveIndexedTriangle)
      {
        vertIndices = primitiveAttribPtr[index].triangleIndex;
      }
      else
      {
        vertIndices = constructUint3(0, 1, 2) + index * 3;
        // for indexed array a non zero stride is assumed
        if (attributePackingInfo.strideIn4Bytes > 0)
        {
          vertIndices = primitiveAttribPtr[index].triangleIndex;
        }
      }

      // get vertex zero and vertex position
      PrimitiveStruct vert0 = primitiveBuffer[vertIndices.x];
      PrimitiveStruct vert1 = primitiveBuffer[vertIndices.y];
      PrimitiveStruct vert2 = primitiveBuffer[vertIndices.z];

      vert0.position = mulMatrixVec(matrix, constructFloat4(vert0.position, 1.f)).xyz;
      vert1.position = mulMatrixVec(matrix, constructFloat4(vert1.position, 1.f)).xyz;
      vert2.position = mulMatrixVec(matrix, constructFloat4(vert2.position, 1.f)).xyz;

      const VertexAttrib vertAttrib0 = vertexAttribPtr[vertIndices.x];
      const VertexAttrib vertAttrib1 = vertexAttribPtr[vertIndices.y];
      const VertexAttrib vertAttrib2 = vertexAttribPtr[vertIndices.z];

      if (primitiveType == PrimitiveIndexedTriangle)
      {
        vertIndices += vertexOffset;
      }
      else
      {
        vert1.position = vert1.position - vert0.position;
        vert2.position = vert2.position - vert0.position;
        vertIndices = vertexOffset + constructUint3(0, 1, 2) + index * 3;
      }

      vert0.identity = primitiveIdentity;
      vert1.identity = primitiveIdentity;
      vert2.identity = primitiveIdentity;

      finalVertexArray[vertIndices.x] = vert0;
      finalVertexArray[vertIndices.y] = vert1;
      finalVertexArray[vertIndices.z] = vert2;

      finalVertexAttributeArray[vertIndices.x] = vertAttrib0;
      finalVertexAttributeArray[vertIndices.y] = vertAttrib1;
      finalVertexAttributeArray[vertIndices.z] = vertAttrib2;

      finalAttributeArray[index + primitiveOffset].quadIndex = constructUint4(vertIndices, -1);
    }
    else if (primitiveType == PrimitiveIndexedQuad)
    {
      uint4 vertIndices;
      {
        vertIndices = primitiveAttribPtr[index].quadIndex;
      }

      // get vertex zero and vertex position
      PrimitiveStruct vert0 = primitiveBuffer[vertIndices.x];
      PrimitiveStruct vert1 = primitiveBuffer[vertIndices.y];
      PrimitiveStruct vert2 = primitiveBuffer[vertIndices.z];
      PrimitiveStruct vert3;

      vert0.position = mulMatrixVec(matrix, constructFloat4(vert0.position, 1.f)).xyz;
      vert1.position = mulMatrixVec(matrix, constructFloat4(vert1.position, 1.f)).xyz;
      vert2.position = mulMatrixVec(matrix, constructFloat4(vert2.position, 1.f)).xyz;

      const VertexAttrib vertAttrib0 = vertexAttribPtr[vertIndices.x];
      const VertexAttrib vertAttrib1 = vertexAttribPtr[vertIndices.y];
      const VertexAttrib vertAttrib2 = vertexAttribPtr[vertIndices.z];
      VertexAttrib vertAttrib3;

      if (vertIndices.w != -1)
      {
        vert3 = primitiveBuffer[vertIndices.w];
        vert3.position = mulMatrixVec(matrix, constructFloat4(vert3.position, 1.f)).xyz;
        vertAttrib3 = vertexAttribPtr[vertIndices.w];
        vertIndices.w += vertexOffset;
        vert3.identity = primitiveIdentity;
      }

      if (primitiveType == PrimitiveIndexedQuad)
      {
        vertIndices.xyz += vertexOffset;
      }

      vert0.identity = primitiveIdentity;
      vert1.identity = primitiveIdentity;
      vert2.identity = primitiveIdentity;

      finalVertexArray[vertIndices.x] = vert0;
      finalVertexArray[vertIndices.y] = vert1;
      finalVertexArray[vertIndices.z] = vert2;

      if (vertIndices.w != -1)
      {
        finalVertexArray[vertIndices.w] = vert3;
        finalVertexAttributeArray[vertIndices.w] = vertAttrib3;
      }

      finalVertexAttributeArray[vertIndices.x] = vertAttrib0;
      finalVertexAttributeArray[vertIndices.y] = vertAttrib1;
      finalVertexAttributeArray[vertIndices.z] = vertAttrib2;

      finalAttributeArray[index + primitiveOffset].quadIndex = vertIndices;
    }
  }
}

#endif
