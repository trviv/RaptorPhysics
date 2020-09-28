#include "AccelerationDataStruct.h"

uint AccelerationDataStruct::accXABComputeUtilId = -1;

//#define DEBUG_ACCELERATION_DATA_STRUCT

uint AccelerationDataStruct::PrimitiveAttributeInfo::bindToShader(ComputeKernel& kernel, uint startIndex)
{
  if (type == PrimitiveSphere)
  {
    kernel.setArg(attributeBuffer[PrimitiveAttributePosition], startIndex);
    kernel.setArg(attributeBuffer[PrimitiveAttributeRadius], startIndex+1);
    kernel.setArg(&attributeInfo[PrimitiveAttributeRadius], startIndex+2);
    return startIndex+3;
  }
  else
  if (type == PrimitiveTriangle)
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
  primStartingOffset.free();
  boundingBoxes.free();
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

  boundingBoxes.create(compute);
  primStartingOffset.create(compute);
  primitiveArray.create(compute);

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

  primitiveCount = 0;
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
  primInfo.count                                        = count;
  primInfo.type                                         = PrimitiveSphere;

  registeredPrimitives.push_back(primInfo);
  primitiveCount += count;
}

void AccelerationDataStruct::registerTriangles(const ComputeMemory* primitiveBuffer, const ComputeMemory* indexBuffer, PackingInfo indexInfo, uint count)
{
  PrimitiveAttributeInfo primInfo;

  primInfo.attributeBuffer[PrimitiveAttributePosition]  = primitiveBuffer;
  primInfo.attributeInfo[PrimitiveAttributePosition]    = PackingInfo();
  primInfo.attributeBuffer[PrimitiveAttributeIndex]     = indexBuffer;
  primInfo.attributeInfo[PrimitiveAttributeIndex]       = indexInfo;
  primInfo.count                                        = count;
  primInfo.type                                         = PrimitiveTriangle;

  registeredPrimitives.push_back(primInfo);
  primitiveCount += count;
}

void AccelerationDataStruct::commit()
{
  primStartingOffset.host()->clear();
  primStartingOffset.host()->push_back((uint)registeredPrimitives.size());

  uint primCount = 0;
  for (const auto& p : registeredPrimitives)
  {
    primCount += p.count;
    primStartingOffset.host()->push_back(primCount);
  }

  primStartingOffset.syncDevice();
  boundingBoxes.resize(primitiveCount, false);
  primitiveArray.resize(primitiveCount, false);
}

void AccelerationDataStruct::fullBuild()
{
  uint primitiveArrayOffset = 0;

  for (auto& prim : registeredPrimitives)
  {
    uint primBatchSize = 8;
    uint primBatchCount = mAlignBy(prim.count, primBatchSize);
    uint primType = prim.type;

    // add to system bounding box
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, primBatchCount);

    createPrimitiveBoundingBoxes.setArg(primitiveArray.device(), 0);
    createPrimitiveBoundingBoxes.setArg(boundingBoxes.device(), 1);
    uint nextBindIndex = prim.bindToShader(createPrimitiveBoundingBoxes, 2);
    createPrimitiveBoundingBoxes.setArg(&primBatchSize, nextBindIndex);
    createPrimitiveBoundingBoxes.setArg(&prim.count, nextBindIndex+1);
    createPrimitiveBoundingBoxes.setArg(&primType, nextBindIndex+2);
    createPrimitiveBoundingBoxes.setArg(&primitiveArrayOffset, nextBindIndex+3);

    compute->execute(createPrimitiveBoundingBoxes, workgroupSize, workgroupCount);

#ifdef DEBUG_ACCELERATION_DATA_STRUCT
    boundingBoxes.syncHost();
    compute->sync();
#endif

    primitiveArrayOffset += prim.count;
  }
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
    uint nextBindIndex = registeredPrimitives[0].bindToShader(intersectionKernel, 4);
    intersectionKernel.setArg(&primitiveCount, nextBindIndex);
    intersectionKernel.setArg(primStartingOffset.device(), nextBindIndex+1);

    compute->execute(intersectionKernel, workgroupSize, workgroupCount);

#ifdef DEBUG_ACCELERATION_DATA_STRUCT
    boundingBoxes.syncHost();
    compute->sync();
#endif
  }
}
