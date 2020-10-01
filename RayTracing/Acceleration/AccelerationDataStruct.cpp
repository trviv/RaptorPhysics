#include "AccelerationDataStruct.h"

uint AccelerationDataStruct::accXABComputeUtilId = -1;
uint AccelerationDataStruct::sortComputeUtilId   = -1;

//#define DEBUG_ACCELERATION_DATA_STRUCT

uint AccelerationDataStruct::PrimitiveAttributeInfo::bindToShader(ComputeKernel& kernel, uint startIndex)
{
  if (primInfo.primType == PrimitiveSphere)
  {
    kernel.setArg(attributeBuffer[PrimitiveAttributePosition], startIndex);
    kernel.setArg(attributeBuffer[PrimitiveAttributeRadius], startIndex+1);
    kernel.setArg(&attributeInfo[PrimitiveAttributeRadius], startIndex+2);
    return startIndex+3;
  }
  else
  if (primInfo.primType == PrimitiveTriangle)
  {
    kernel.setArg(attributeBuffer[PrimitiveAttributePosition], startIndex);
    if (attributeInfo[PrimitiveAttributeIndex].strideIn4Bytes)
    {
      kernel.setArg(attributeBuffer[PrimitiveAttributeIndex], startIndex+1);
    }
    else
    {
      kernel.setArg(attributeBuffer[PrimitiveAttributePosition], startIndex+1);
    }
    kernel.setArg(&attributeInfo[PrimitiveAttributeIndex], startIndex+2);
    return startIndex+3;
  }

  return startIndex;
}

AccelerationDataStruct::AccelerationDataStruct()
{
}

AccelerationDataStruct::~AccelerationDataStruct()
{
}

void AccelerationDataStruct::create(ComputeInterface* compute)
{
  this->compute = compute;
  includeFiles.push_back("ComputeHeader.shader");
  includeFiles.push_back("ComputeShared.h");
  includeFiles.push_back("RayStructs.h");
  includeFiles.push_back("HitStructs.h");
  includeFiles.push_back("RayTracingStruct.h");

  registerShader(compute, "AccelerationDataStructCreate.shader", NULL, NULL);

  createPrimitiveBoundingBoxes = programs[0].createKernel("createPrimitiveBoundingBoxes");
  assignMortonCode = programs[0].createKernel("assignMortonCode");

  for (int r=0; r<RayStructTypeMax; r++)
  {
    for (int h=0; h<HitStructTypeMax; h++)
    {
      const vector<string> oldType = {"RayStruct", "HitStruct"};
      const vector<string> newType = {getRayStructName((RayStructType)r), getHitStructName((HitStructType)h)};
      registerShader(compute, "AccelerationDataStructTraverse.shader", &oldType, &newType);
      intersectRayKernels[r][h] = programs.back().createKernel("intersectRays");
    }
  }

  systemSettings.create(compute);
  boundingBoxes.create(compute);
  vertexArray.create(compute);
  primitiveLeafData.create(compute);
  primitiveLeafDataSorted.create(compute);

  for (auto& i : registeredPrimitives)
  {
    i.clear();
  }

  map<ComputeUtilKey, string> lbvhXABSetting;
  lbvhXABSetting[ComputeUtilBatchSize] = "1";
  lbvhXABSetting[ComputeUtilStructType] = "XAB";
  lbvhXABSetting[ComputeUtilStructSize] = "32";
  lbvhXABSetting[ComputeUtilOnlyReduce] = "1";
  lbvhXABSetting[ComputeUtilCustomAddFunction] = "mergeXAB";
  lbvhXABSetting[ComputeUtilCustomDivFunction] = "divXAB";
  lbvhXABSetting[ComputeUtilCustomCopyFunction] = "copyXAB";
  lbvhXABSetting[ComputeUtilCustomClearFunction] = "clearXAB";
  lbvhXABSetting[ComputeUtilCustomReduceFunction] = "reduceXAB";
  lbvhXABSetting[ComputeUtilSkipParallelPrimitives] = "1";
  accXABComputeUtilId = ComputeUtil::create(compute, lbvhXABSetting, NULL);

  map<ComputeUtilKey, string> lbvhSortSetting;
  lbvhSortSetting[ComputeUtilStructType] = "uint";
  lbvhSortSetting[ComputeUtilStructTypeIntegral] = "1";
  sortComputeUtilId = ComputeUtil::create(compute, lbvhSortSetting, NULL);

  primitiveCount = 0;
  vertexCount    = 0;
}

uint AccelerationDataStruct::getPrimCount()const
{
  return primitiveCount;
}

void AccelerationDataStruct::registerSpheres(const ComputeMemory* primitiveBuffer, const ComputeMemory* radiusBuffer, PackingInfo radiusInfo, uint count)
{
  PrimitiveAttributeInfo primInfo;

  primInfo.attributeBuffer[PrimitiveAttributePosition]  = primitiveBuffer;
  primInfo.attributeInfo[PrimitiveAttributePosition]    = PackingInfo();
  primInfo.attributeBuffer[PrimitiveAttributeRadius]    = radiusBuffer;
  primInfo.attributeInfo[PrimitiveAttributeRadius]      = radiusInfo;
  primInfo.primInfo.primType    = PrimitiveSphere;
  primInfo.primInfo.indexCount  = count;
  primInfo.primInfo.vertexCount = count;

  registeredPrimitives[PrimitiveSphere].push_back(primInfo);
}

void AccelerationDataStruct::registerTriangles(const ComputeMemory* primitiveBuffer, const ComputeMemory* indexBuffer, PackingInfo indexInfo, uint count)
{
  PrimitiveAttributeInfo primInfo;

  primInfo.attributeBuffer[PrimitiveAttributePosition]  = primitiveBuffer;
  primInfo.attributeInfo[PrimitiveAttributePosition]    = PackingInfo();
  primInfo.attributeBuffer[PrimitiveAttributeIndex]     = indexBuffer;
  primInfo.attributeInfo[PrimitiveAttributeIndex]       = indexInfo;
  primInfo.primInfo.primType    = PrimitiveTriangle;
  primInfo.primInfo.indexCount  = count;
  primInfo.primInfo.vertexCount = count * 3;

  registeredPrimitives[PrimitiveTriangle].push_back(primInfo);
}

void AccelerationDataStruct::commit()
{
  systemSettings.host()->resize(1);

  uint indexOffset  = 0;
  uint vertexOffset = 0;

  for (uint i=0; i<RTPrimitiveCount; i++)
  {
    EncodedPrimitiveInfo primInfo;

    setPrimitiveType(primInfo,         (RTPrimitiveType)i);
    setPrimitiveIndexOffset(primInfo,  indexOffset);
    setPrimitiveVertexOffset(primInfo, vertexOffset);
    systemSettings.host()->at(0).globalOffsets[i] = primInfo;

    for (const auto& p : registeredPrimitives[i])
    {
      indexOffset  += p.primInfo.indexCount;
      vertexOffset += p.primInfo.vertexCount;
    }
  }

  primitiveCount = indexOffset;
  vertexCount    = vertexOffset;

  systemSettings.resize(1, false);
  vertexArray.resize(vertexCount, false);
  boundingBoxes.resize(primitiveCount, false);
  primitiveLeafData.resize(primitiveCount, false);
  primitiveLeafDataSorted.resize(primitiveCount, false);

  systemSettings.syncHost();
}

void AccelerationDataStruct::fullBuild()
{
  uint indexOffset = 0;
  uint vertexOffset = 0;
  uint primBatchSize = 8;

  for (auto& rp : registeredPrimitives)
  {
    for (uint i=0; i<rp.size(); i++)
    {
      auto& prim = rp[i];
      uint primBatchCount = mAlignBy(prim.primInfo.indexCount, primBatchSize);
      uint primType = prim.primInfo.primType;

      size_t workgroupSize[3], workgroupCount[3];
      compute->configureSize(workgroupSize, workgroupCount, primBatchCount);

      createPrimitiveBoundingBoxes.setArg(vertexArray.device(), 0);
      createPrimitiveBoundingBoxes.setArg(boundingBoxes.device(), 1);
      uint nextBindIndex = prim.bindToShader(createPrimitiveBoundingBoxes, 2);
      createPrimitiveBoundingBoxes.setArg(&primBatchSize, nextBindIndex);
      createPrimitiveBoundingBoxes.setArg(&prim.primInfo.indexCount, nextBindIndex+1);
      createPrimitiveBoundingBoxes.setArg(&primType, nextBindIndex+2);
      createPrimitiveBoundingBoxes.setArg(&indexOffset, nextBindIndex+3);
      createPrimitiveBoundingBoxes.setArg(&vertexOffset, nextBindIndex+4);

      compute->execute(createPrimitiveBoundingBoxes, workgroupSize, workgroupCount);

#ifdef DEBUG_ACCELERATION_DATA_STRUCT
      boundingBoxes.syncHost();
      compute->sync();
#endif

      indexOffset += prim.primInfo.indexCount;
      vertexOffset += prim.primInfo.vertexCount;
    }
  }

  // find bounding box for the simulation space
  ComputeUtil::get(accXABComputeUtilId)->sum1D(compute, systemSettings.device(), boundingBoxes.device(), primitiveCount);

#ifdef DEBUG_ACCELERATION_DATA_STRUCT
  systemSettings.syncHost();
  compute->sync();
#endif

  {
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, primitiveCount);

    // assign morton code to the particle bounding boxes
    ComputeMemory* buffers[] = {
      primitiveLeafData.device(),
      vertexArray.device(),
      systemSettings.device()
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    assignMortonCode.setArgs(buffers, bufferCount);
    assignMortonCode.setArg(&primBatchSize, bufferCount);
    assignMortonCode.setArg(&primitiveCount, bufferCount+1);

    compute->execute(assignMortonCode, workgroupSize, workgroupCount);
  }

#ifdef DEBUG_ACCELERATION_DATA_STRUCT
  primitiveLeafData.syncHost();
  compute->sync();
#endif

  ComputeUtil::get(sortComputeUtilId)->radixSort32Bit(compute, primitiveLeafDataSorted.device(), primitiveLeafData.device(), primitiveCount);

#ifdef DEBUG_ACCELERATION_DATA_STRUCT
  primitiveLeafDataSorted.syncHost();
  compute->sync();
#endif
}

void AccelerationDataStruct::intersectRays(ComputeMemory* hits, HitStructType hitType, ComputeMemory* rays, RayStructType rayType, uint rayCount)
{
  {
    ComputeKernel& intersectionKernel = intersectRayKernels[rayType][hitType];

    // add to system bounding box
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, rayCount);

    intersectionKernel.setArg(hits, 0);
    intersectionKernel.setArg(rays, 1);
    intersectionKernel.setArg(&rayCount, 2);
    intersectionKernel.setArg(boundingBoxes.device(), 3);
    uint nextBindIndex = registeredPrimitives[0][0].bindToShader(intersectionKernel, 4);
    intersectionKernel.setArg(&primitiveCount, nextBindIndex);
    intersectionKernel.setArg(systemSettings.device(), nextBindIndex+1);

    compute->execute(intersectionKernel, workgroupSize, workgroupCount);

#ifdef DEBUG_ACCELERATION_DATA_STRUCT
    boundingBoxes.syncHost();
    compute->sync();
#endif
  }
}
