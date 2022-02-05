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

void BoundingVolumeHierarchyADS::initializeData()
{
  AccelerationDataStruct::initializeData();
  visitedInternalNodes.create(compute);
  leafParentNodeIndices.create(compute);
  nodeParentNodeIndices.create(compute);
  treeNodeBoundingBoxes.create(compute);
  treeInternalNodes.create(compute);
  primitiveLeafData.create(compute);
  primitiveLeafDataSorted.create(compute);
}

void BoundingVolumeHierarchyADS::updatePointers()
{
  pointerLeafParentNodeIndices = leafParentNodeIndices.device();
  pointerNodeParentNodeIndices = nodeParentNodeIndices.device();
  pointerLeafNodeBoundingBoxes = leafNodeBoundingBoxes.device();
  pointerTreeNodeBoundingBoxes = treeNodeBoundingBoxes.device();
  pointerTreeInternalNodes     = treeInternalNodes.device();
}

void BoundingVolumeHierarchyADS::registerCreateShaders(const vector<string>* oldType, const vector<string>* newType)
{
  includeFiles.push_back("ComputeHeader.shader");
  includeFiles.push_back("ComputeShared.h");
  includeFiles.push_back("RayTracingStruct.h");
  includeFiles.push_back("AccelerationDataStructCreate.shader");

  registerShader(compute, "BoundingVolumeHierarchyADSCreate.shader", oldType, newType);

  createPrimitiveBoundingBoxes = programs[0].createKernel("createPrimitiveBoundingBoxes");
  assignMortonCode             = programs[0].createKernel("assignMortonCode");
  constructBinaryTree          = programs[0].createKernel("constructBinaryTree");
  constructTreeBoundingBox     = programs[0].createKernel("constructTreeBoundingBox");
}

void BoundingVolumeHierarchyADS::registerTraverseShaders(const vector<string>* oldTypeArg, const vector<string>* newTypeArg)
{
  includeFiles.push_back("RayStructs.h");
  includeFiles.push_back("HitStructs.h");
  includeFiles.push_back("BoundingVolumeHierarchyADSCreate.shader");
  includeFiles.push_back("AccelerationDataStructTraverse.shader");

  string intersectFunctionName = "intersectRaysBVH";
  string traverseShaderProgram = "BoundingVolumeHierarchyADSTraverse.shader";

  if (oldTypeArg)
  {
    auto functionIter = find(oldTypeArg->begin(), oldTypeArg->end(), string("BVH_ADS_INTERSECT_RAY_BVH_FUNCTION"));
    if (functionIter != oldTypeArg->end())
    {
      intersectFunctionName = newTypeArg->at(functionIter - oldTypeArg->begin());
    }

    functionIter = find(oldTypeArg->begin(), oldTypeArg->end(), string("ADS_TRAVERSAL_SHADER_PROGRAM"));
    if (functionIter != oldTypeArg->end())
    {
      traverseShaderProgram = newTypeArg->at(functionIter - oldTypeArg->begin());
    }
  }

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

        if (oldTypeArg) oldType.insert(oldType.end(), oldTypeArg->begin(), oldTypeArg->end());
        if (newTypeArg) newType.insert(newType.end(), newTypeArg->begin(), newTypeArg->end());

        registerShader(compute, traverseShaderProgram.c_str(), &oldType, &newType);
        intersectRayKernels[i][r][h] = programs.back().createKernel(intersectFunctionName.c_str());
      }
    }
  }
}

void BoundingVolumeHierarchyADS::registerResources(ComputeKernel& kernel)const
{
  kernel.registerResource(treeInternalNodes.device());
  kernel.registerResource(leafParentNodeIndices.device());
  kernel.registerResource(nodeParentNodeIndices.device());
  kernel.registerResource(leafNodeBoundingBoxes.device());
  kernel.registerResource(treeNodeBoundingBoxes.device());
  kernel.registerResource(pointerVertexArray);
  kernel.registerResource(pointerAttributeArray);
  kernel.registerResource(pointerSystemSettings);
}

void BoundingVolumeHierarchyADS::bindBuffers(const ComputeMemory* vertexArray, const ComputeMemory* attributeArray,
                                             DeviceArray<RTSystemSettings>* systemSettings)
{
  AccelerationDataStruct::bindBuffers(vertexArray, attributeArray, systemSettings);
  visitedInternalNodes.resize(primitiveCount, false);
  leafParentNodeIndices.resize(primitiveCount, false);
  nodeParentNodeIndices.resize(primitiveCount, false);
  treeNodeBoundingBoxes.resize(primitiveCount - 1, false);
  treeInternalNodes.resize(primitiveCount - 1, false);
  primitiveLeafData.resize(primitiveCount, false);
  primitiveLeafDataSorted.resize(primitiveCount, false);
}

void BoundingVolumeHierarchyADS::fullBuild()
{
  if (!needsRebuild) return;

  updatePointers();

  uint primBatchSize = 8;
  uint primBatchCount = mAlignBy(primitiveCount, primBatchSize);

  {
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, primBatchCount);

    createPrimitiveBoundingBoxes.setArg(pointerLeafNodeBoundingBoxes, 0);
    createPrimitiveBoundingBoxes.setArg(pointerVertexArray, 1);
    createPrimitiveBoundingBoxes.setArg(pointerAttributeArray, 2);
    createPrimitiveBoundingBoxes.setArg(pointerSystemSettings, 3);
    createPrimitiveBoundingBoxes.setArg(&primBatchSize, 4);
    createPrimitiveBoundingBoxes.setArg(&primitiveCount, 5);

    compute->execute(createPrimitiveBoundingBoxes, workgroupSize, workgroupCount);

#ifdef DEBUG_BVH_ADS
    leafNodeBoundingBoxes.syncHost();
    compute->sync();
#endif
  }

  // find bounding box for the simulation space
  ComputeUtil::get(accXABComputeUtilId)->sum1D(compute, pointerSystemSettings, pointerLeafNodeBoundingBoxes, primitiveCount);

#ifdef DEBUG_BVH_ADS
  compute->sync();
#endif

  {
    uint primBatchCount = mAlignBy(primitiveCount, primBatchSize);
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, primBatchCount);

    // assign morton code to the particle bounding boxes
    assignMortonCode.setArg(primitiveLeafData.device(), 0);
    assignMortonCode.setArg(pointerVertexArray, 1);
    assignMortonCode.setArg(pointerAttributeArray, 2);
    assignMortonCode.setArg(pointerSystemSettings, 3);
    assignMortonCode.setArg(&primBatchSize, 4);
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
    constructBinaryTree.setArg(pointerTreeInternalNodes, 0);
    constructBinaryTree.setArg(visitedInternalNodes.device(), 1);
    constructBinaryTree.setArg(pointerLeafParentNodeIndices, 2);
    constructBinaryTree.setArg(pointerNodeParentNodeIndices, 3);
    constructBinaryTree.setArg(primitiveLeafDataSorted.device(), 4);
    constructBinaryTree.setArg<uint>(&primitiveCount, 5);

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
    constructTreeBoundingBox.setArg(pointerTreeNodeBoundingBoxes, 0);
    constructTreeBoundingBox.setArg(visitedInternalNodes.device(), 1);
    constructTreeBoundingBox.setArg(pointerTreeInternalNodes, 2);
    constructTreeBoundingBox.setArg(pointerLeafParentNodeIndices, 3);
    constructTreeBoundingBox.setArg(pointerNodeParentNodeIndices, 4);
    constructTreeBoundingBox.setArg(pointerLeafNodeBoundingBoxes, 5);
    constructTreeBoundingBox.setArg<uint>(&primitiveCount, 6);

    compute->execute(constructTreeBoundingBox, workgroupSize, workgroupCount);
  }

#ifdef DEBUG_BVH_ADS
  treeNodeBoundingBoxes.syncHost();
  visitedInternalNodes.syncHost();
  compute->sync();
#endif

  needsRebuild = false;
}

void BoundingVolumeHierarchyADS::intersectRays(ComputeMemory* hits, HitStructType hitType, const ComputeMemory* rays, RayStructType rayType,
                                               uint rayCount, IntersectionType intersectionType)
{
  validateBuild();
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
    intersectionKernel.setArg(pointerVertexArray, 3);
    intersectionKernel.setArg(pointerAttributeArray, 4);
    intersectionKernel.setArg(pointerTreeInternalNodes, 5);
    intersectionKernel.setArg(pointerLeafParentNodeIndices, 6);
    intersectionKernel.setArg(pointerNodeParentNodeIndices, 7);
    intersectionKernel.setArg(pointerLeafNodeBoundingBoxes, 8);
    intersectionKernel.setArg(pointerTreeNodeBoundingBoxes, 9);
    intersectionKernel.setArg(pointerSystemSettings, 10);
    intersectionKernel.setArg(&primitiveCount, 11);
    intersectionKernel.setArg(visitedInternalNodes.device(), 12);
    intersectionKernel.setSharedMemArg(4 * max(workgroupSize[0] * workgroupSize[1] * workgroupSize[2] * RAY_TRAVERSAL_SHARED_MEMORY_INDEX_STRIDE, (size_t)4), 13);

    compute->execute(intersectionKernel, workgroupSize, workgroupCount);

#ifdef DEBUG_BVH_ADS
    leafNodeBoundingBoxes.syncHost();
    compute->sync();
#endif
  }
}

void BoundingVolumeHierarchyADS::intersectRays(ComputeMemory* hits, HitStructType hitType, const ComputeMemory* rays, RayStructType rayType,
                                               const ComputeMemory* rayCount, IntersectionType intersectionType)
{
  validateBuild();
  {
    size_t workgroupSize[3] = {compute->maxThreadsPerGroup() * BVH_ADS_PERSISTENT_MULTIPLIER, 1, 1};
    ComputeUtil::get(sortComputeUtilId)->configureWorkgroupCount(compute, workgroupCount.device(), rayCount, workgroupSize);
    workgroupSize[0] /= BVH_ADS_PERSISTENT_MULTIPLIER;

    ComputeKernel& intersectionKernel = intersectRayKernels[intersectionType][rayType][hitType];

    intersectionKernel.setArg(hits, 0);
    intersectionKernel.setArg(rays, 1);
    intersectionKernel.setArg(rayCount, 2);
    intersectionKernel.setArg(pointerVertexArray, 3);
    intersectionKernel.setArg(pointerAttributeArray, 4);
    intersectionKernel.setArg(pointerTreeInternalNodes, 5);
    intersectionKernel.setArg(pointerLeafParentNodeIndices, 6);
    intersectionKernel.setArg(pointerNodeParentNodeIndices, 7);
    intersectionKernel.setArg(pointerLeafNodeBoundingBoxes, 8);
    intersectionKernel.setArg(pointerTreeNodeBoundingBoxes, 9);
    intersectionKernel.setArg(pointerSystemSettings, 10);
    intersectionKernel.setArg(&primitiveCount, 11);
    intersectionKernel.setArg(workgroupCount.device(), 12);
    intersectionKernel.setSharedMemArg(4 * max(workgroupSize[0] * workgroupSize[1] * workgroupSize[2] * RAY_TRAVERSAL_SHARED_MEMORY_INDEX_STRIDE, (size_t)4), 13);

    compute->execute(intersectionKernel, workgroupSize, workgroupCount.device(), 0);

#ifdef DEBUG_BVH_ADS
    leafNodeBoundingBoxes.syncHost();
    compute->sync();
#endif
  }
}
