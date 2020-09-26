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
}

void AccelerationDataStruct::fullBuild()
{
  for (auto& prim : registeredPrimitives)
  {
    uint primBatchSize = 8;
    uint primBatchCount = mAlignBy(prim.count, primBatchSize);

    // add to system bounding box
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, primBatchCount);

    createPrimitiveBoundingBoxes.setArg(boundingBoxes.device(), 0);
    uint nextBindIndex = prim.bindToShader(createPrimitiveBoundingBoxes, 1);
    createPrimitiveBoundingBoxes.setArg(&primBatchSize, nextBindIndex);
    createPrimitiveBoundingBoxes.setArg(&prim.count, nextBindIndex+1);
    createPrimitiveBoundingBoxes.setArg(primStartingOffset.device(), nextBindIndex+2);

    compute->execute(createPrimitiveBoundingBoxes, workgroupSize, workgroupCount);

#ifdef DEBUG_ACCELERATION_DATA_STRUCT
    boundingBoxes.syncHost();
    compute->sync();
#endif
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
    intersectionKernel.setArg(&registeredPrimitives[0].count, nextBindIndex);
    intersectionKernel.setArg(primStartingOffset.device(), nextBindIndex+1);

    compute->execute(intersectionKernel, workgroupSize, workgroupCount);

#ifdef DEBUG_ACCELERATION_DATA_STRUCT
    boundingBoxes.syncHost();
    compute->sync();
#endif
  }
}
