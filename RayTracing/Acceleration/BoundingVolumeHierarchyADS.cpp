#include "BoundingVolumeHierarchyADS.h"

//#define DEBUG_BVH_ADS
#define RAY_TRAVERSAL_BVH_MAX_LEAFS 5
#define BVH_ADS_PERSISTENT_MULTIPLIER 1
#define RAY_TRAVERSAL_SHARED_MEMORY_INDEX_STRIDE RAY_TRAVERSAL_BVH_MAX_LEAFS

BoundingVolumeHierarchyADS::BoundingVolumeHierarchyADS()
{
}

BoundingVolumeHierarchyADS::~BoundingVolumeHierarchyADS()
{
}

void BoundingVolumeHierarchyADS::create(ComputeInterface* compute)
{
  this->compute = compute;
  includeFiles.push_back("ComputeHeader.shader");
  includeFiles.push_back("ComputeShared.h");
  includeFiles.push_back("RayTracingStruct.h");
  includeFiles.push_back("AccelerationDataStructCreate.shader");

  registerShader(compute, "BoundingVolumeHierarchyADSCreate.shader", NULL, NULL);

  createPrimitiveBoundingBoxes = programs[0].createKernel("createPrimitiveBoundingBoxes");
  assignMortonCode             = programs[0].createKernel("assignMortonCode");
  constructBinaryTree          = programs[0].createKernel("constructBinaryTree");
  constructTreeBoundingBox     = programs[0].createKernel("constructTreeBoundingBox");

  includeFiles.push_back("RayStructs.h");
  includeFiles.push_back("HitStructs.h");
  includeFiles.push_back("BoundingVolumeHierarchyADSCreate.shader");
  includeFiles.push_back("AccelerationDataStructTraverse.shader");

  for (int i=0; i<IntersectionTypeMax; i++)
  {
    for (int r=0; r<RayStructTypeMax; r++)
    {
      for (int h=0; h<HitStructTypeMax; h++)
      {
        vector<string> oldType = {getIntersectionTypeName((IntersectionType)i), "RayStruct", "HitStruct", "BVH_ADS_PERSISTENT_MULTIPLIER",
          "RAY_TRAVERSAL_BVH_MAX_LEAFS", "RAY_TRAVERSAL_SHARED_MEMORY_INDEX_STRIDE"};
        vector<string> newType = {"", getRayStructName((RayStructType)r), getHitStructName((HitStructType)h), to_string(BVH_ADS_PERSISTENT_MULTIPLIER),
          to_string(RAY_TRAVERSAL_BVH_MAX_LEAFS), to_string(RAY_TRAVERSAL_SHARED_MEMORY_INDEX_STRIDE)};
//        oldType.push_back("STACKLESS_TRAVERSE_EARLY_CHILD");
//        newType.push_back("");
        getRayStructDefines(oldType, newType, (RayStructType)r);
        getHitStructDefines(oldType, newType, (HitStructType)h);
        registerShader(compute, "BoundingVolumeHierarchyADSTraverse.shader", &oldType, &newType);
        intersectRayKernels[i][r][h] = programs.back().createKernel("intersectRaysBVH");
      }
    }
  }

  boundingBoxes.create(compute);
  visitedInternalNodes.create(compute);
  leafParentNodeIndices.create(compute);
  nodeParentNodeIndices.create(compute);
  treeInternalNodeBoundingBoxes.create(compute);
  treeInternalNodes.create(compute);
  primitiveLeafData.create(compute);
  primitiveLeafDataSorted.create(compute);

  accXABComputeUtilId = ComputeUtil::getXABUtil(compute);
  sortComputeUtilId   = ComputeUtil::getUIntUtil(compute);

  primitiveCount  = 0;
  vertexCount     = 0;

  workgroupCount.create(compute);
  workgroupCount.resize(4, false);
}

void BoundingVolumeHierarchyADS::bindBuffers(const ComputeMemory* vertexArray, const ComputeMemory* attributeArray,
                                             DeviceArray<RTSystemSettings>* systemSettings)
{
  AccelerationDataStruct::bindBuffers(vertexArray, attributeArray, systemSettings);
  visitedInternalNodes.resize(primitiveCount, false);
  leafParentNodeIndices.resize(primitiveCount, false);
  nodeParentNodeIndices.resize(primitiveCount, false);
  treeInternalNodeBoundingBoxes.resize(primitiveCount - 1, false);
  treeInternalNodes.resize(primitiveCount - 1, false);
  primitiveLeafData.resize(primitiveCount, false);
  primitiveLeafDataSorted.resize(primitiveCount, false);
}

void BoundingVolumeHierarchyADS::fullBuild()
{
  uint primBatchSize = 8;
  uint primBatchCount = mAlignBy(primitiveCount, primBatchSize);

  {
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, primBatchCount);

    createPrimitiveBoundingBoxes.setArg(boundingBoxes.device(),   0);
    createPrimitiveBoundingBoxes.setArg(vertexArray,              1);
    createPrimitiveBoundingBoxes.setArg(attributeArray,           2);
    createPrimitiveBoundingBoxes.setArg(systemSettings->device(), 3);
    createPrimitiveBoundingBoxes.setArg(&primBatchSize,           4);
    createPrimitiveBoundingBoxes.setArg(&primitiveCount,          5);

    compute->execute(createPrimitiveBoundingBoxes, workgroupSize, workgroupCount);

#ifdef DEBUG_BVH_ADS
    boundingBoxes.syncHost();
    compute->sync();
#endif
  }

  // find bounding box for the simulation space
  ComputeUtil::get(accXABComputeUtilId)->sum1D(compute, systemSettings->device(), boundingBoxes.device(), primitiveCount);

#ifdef DEBUG_BVH_ADS
  systemSettings->syncHost();
  compute->sync();
#endif

  {
    uint primBatchCount = mAlignBy(primitiveCount, primBatchSize);
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, primBatchCount);

    // assign morton code to the particle bounding boxes
    assignMortonCode.setArg(primitiveLeafData.device(), 0);
    assignMortonCode.setArg(vertexArray,                1);
    assignMortonCode.setArg(attributeArray,             2);
    assignMortonCode.setArg(systemSettings->device(),   3);
    assignMortonCode.setArg(&primBatchSize,  4);
    assignMortonCode.setArg(&primitiveCount, 5);

    compute->execute(assignMortonCode, workgroupSize, workgroupCount);
  }

#ifdef DEBUG_BVH_ADS
  primitiveLeafData.syncHost();
  compute->sync();
#endif

  ComputeUtil::get(sortComputeUtilId)->radixSort32Bit(compute, primitiveLeafDataSorted.device(), primitiveLeafData.device(), primitiveCount);

#ifdef DEBUG_BVH_ADS
  primitiveLeafDataSorted.syncHost();
  compute->sync();
#endif

  {
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, primitiveCount);

    // create binary radix tree
    ComputeMemory* buffers[] = {
      treeInternalNodes.device(),
      visitedInternalNodes.device(),
      leafParentNodeIndices.device(),
      nodeParentNodeIndices.device(),
      primitiveLeafDataSorted.device()
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    constructBinaryTree.setArgs(buffers, bufferCount);
    constructBinaryTree.setArg<uint>(&primitiveCount, bufferCount);

    compute->execute(constructBinaryTree, workgroupSize, workgroupCount);
  }

#ifdef DEBUG_BVH_ADS
  treeInternalNodes.syncHost();
  leafParentNodeIndices.syncHost();
  nodeParentNodeIndices.syncHost();
  compute->sync();
#endif

  {
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, primitiveCount);

    // calculate bounding boxes for the tree
    ComputeMemory* buffers[] = {
      treeInternalNodeBoundingBoxes.device(),
      visitedInternalNodes.device(),
      treeInternalNodes.device(),
      leafParentNodeIndices.device(),
      nodeParentNodeIndices.device(),
      boundingBoxes.device()
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    constructTreeBoundingBox.setArgs(buffers, bufferCount);
    constructTreeBoundingBox.setArg<uint>(&primitiveCount, bufferCount);

    compute->execute(constructTreeBoundingBox, workgroupSize, workgroupCount);
  }

#ifdef DEBUG_BVH_ADS
  treeInternalNodeBoundingBoxes.syncHost();
  visitedInternalNodes.syncHost();
  compute->sync();
#endif
}

void BoundingVolumeHierarchyADS::intersectRays(ComputeMemory* hits, HitStructType hitType, const ComputeMemory* rays, RayStructType rayType,
                                               uint rayCount, IntersectionType intersectionType)
{
  {
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, mAlignBy(rayCount, BVH_ADS_PERSISTENT_MULTIPLIER));
#if BVH_ADS_PERSISTENT_MULTIPLIER > 1
    ComputeUtil::get(sortComputeUtilId)->clearBuffer(compute, visitedInternalNodes.device(), 1);
#endif

    ComputeKernel& intersectionKernel = intersectRayKernels[intersectionType][rayType][hitType];

    intersectionKernel.setArg(hits, 0);
    intersectionKernel.setArg(rays, 1);
    intersectionKernel.setArg(&rayCount, 2);
    intersectionKernel.setArg(vertexArray, 3);
    intersectionKernel.setArg(attributeArray, 4);
    intersectionKernel.setArg(treeInternalNodes.device(),     5);
    intersectionKernel.setArg(leafParentNodeIndices.device(), 6);
    intersectionKernel.setArg(nodeParentNodeIndices.device(), 7);
    intersectionKernel.setArg(boundingBoxes.device(), 8);
    intersectionKernel.setArg(treeInternalNodeBoundingBoxes.device(), 9);
    intersectionKernel.setArg(systemSettings->device(), 10);
    intersectionKernel.setArg(&primitiveCount,          11);
    intersectionKernel.setArg(visitedInternalNodes.device(),  12);
    intersectionKernel.setSharedMemArg(4 * max(workgroupSize[0] * workgroupSize[1] * workgroupSize[2] * RAY_TRAVERSAL_SHARED_MEMORY_INDEX_STRIDE, (size_t)4), 13);

    compute->execute(intersectionKernel, workgroupSize, workgroupCount);

#ifdef DEBUG_BVH_ADS
    boundingBoxes.syncHost();
    compute->sync();
#endif
  }
}

void BoundingVolumeHierarchyADS::intersectRays(ComputeMemory* hits, HitStructType hitType, const ComputeMemory* rays, RayStructType rayType,
                                               const ComputeMemory* rayCount, IntersectionType intersectionType)
{
  {
    size_t workgroupSize[3] = {compute->maxThreadsPerGroup() * BVH_ADS_PERSISTENT_MULTIPLIER, 1, 1};
    ComputeUtil::get(sortComputeUtilId)->configureWorkgroupCount(compute, workgroupCount.device(), rayCount, workgroupSize);
    workgroupSize[0] /= BVH_ADS_PERSISTENT_MULTIPLIER;

    ComputeKernel& intersectionKernel = intersectRayKernels[intersectionType][rayType][hitType];

    intersectionKernel.setArg(hits, 0);
    intersectionKernel.setArg(rays, 1);
    intersectionKernel.setArg(rayCount, 2);
    intersectionKernel.setArg(vertexArray, 3);
    intersectionKernel.setArg(attributeArray, 4);
    intersectionKernel.setArg(treeInternalNodes.device(),     5);
    intersectionKernel.setArg(leafParentNodeIndices.device(), 6);
    intersectionKernel.setArg(nodeParentNodeIndices.device(), 7);
    intersectionKernel.setArg(boundingBoxes.device(), 8);
    intersectionKernel.setArg(treeInternalNodeBoundingBoxes.device(), 9);
    intersectionKernel.setArg(systemSettings->device(), 10);
    intersectionKernel.setArg(&primitiveCount,          11);
    intersectionKernel.setArg(workgroupCount.device(),  12);
    intersectionKernel.setSharedMemArg(4 * max(workgroupSize[0] * workgroupSize[1] * workgroupSize[2] * RAY_TRAVERSAL_SHARED_MEMORY_INDEX_STRIDE, (size_t)4), 13);

    compute->execute(intersectionKernel, workgroupSize, workgroupCount.device(), 0);

#ifdef DEBUG_BVH_ADS
    boundingBoxes.syncHost();
    compute->sync();
#endif
  }
}
