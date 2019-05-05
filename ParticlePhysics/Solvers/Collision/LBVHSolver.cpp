#include "LBVHSolver.h"

//#define DEBUG_LBVH_SOLVER

#define LBVH_COLLISION_SOLVER_CREATE_BOUNDING_BOX 0
#define LBVH_COLLISION_SOLVER_ASSIGN_MORTON_CODE  1
#define LBVH_COLLISION_SOLVER_CREATE_BINARY_TREE  2
#define LBVH_COLLISION_SOLVER_TREE_BOUNDING_BOX   3

static uint lbvhXABComputeUtilId;
static uint lbvhSortComputeUtilId;

LBVHSolver::~LBVHSolver()
{
}

DeviceArray<XAB>* LBVHSolver::getBoundingBoxes()
{
  return &treeInternalNodeBoundingBoxes;
}

void LBVHSolver::init(ComputeInterface* compute, SharedAllocator* allocator)
{
  this->compute = compute;
  this->allocator = allocator;

  includeFiles.push_back("ComputeHeader.shader");
  includeFiles.push_back("ComputeShared.h");
  includeFiles.push_back("ParticleStruct.h");
  includeFiles.push_back("CollisionSolverShared.h");

  registerShader(compute, "LBVHSolver.shader", NULL, NULL);
  kernels.push_back(programs[0].createKernel("createBoundingBoxes"));
  kernels.push_back(programs[0].createKernel("assignMortonCode"));
  kernels.push_back(programs[0].createKernel("constructBinaryTree"));
  kernels.push_back(programs[0].createKernel("constructTreeBoundingBox"));

  solverHeap = new ComputeHeap(compute);

#ifdef DEBUG_LBVH_SOLVER
  treeInternalNodes.create(compute, NULL, true);
  particleLeafData.create(compute, NULL, true);
  particleLeafDataSorted.create(compute, NULL, true);
  visitedInternalNodes.create(compute, NULL, true);
  leafParentNodeIndices.create(compute, NULL, true);
  systemBoundingBox.create(compute, NULL, true);
  particleBoundingBoxes.create(compute, NULL, true);
#else
  treeInternalNodes.create(compute, NULL);
  particleLeafData.create(compute, NULL);
  particleLeafDataSorted.create(compute, NULL);
  visitedInternalNodes.create(compute, NULL);
  leafParentNodeIndices.create(compute, NULL);
  systemBoundingBox.create(compute, NULL);
  particleBoundingBoxes.create(compute, NULL);
#endif

  treeInternalNodeBoundingBoxes.create(compute, NULL, true);

  systemBoundingBox.resize(1, false);

  vector<string> utilInclude;
  utilInclude.push_back("ParticleStruct.h");
  utilInclude.push_back("CollisionSolverShared.h");

  map<ComputeUtilKey, string> lbvhXABSetting;
  lbvhXABSetting[ComputeUtilBatchSize] = "1";
  lbvhXABSetting[ComputeUtilStructType] = "XAB";
  lbvhXABSetting[ComputeUtilOnlyReduce] = "1";
  lbvhXABSetting[ComputeUtilMaxWorkgroupSize] = "512";
  lbvhXABSetting[ComputeUtilCustomAddFunction] = "mergeXAB";
  lbvhXABSetting[ComputeUtilCustomDivFunction] = "divXAB";
  lbvhXABSetting[ComputeUtilCustomClearFunction] = "clearXAB";
  lbvhXABSetting[ComputeUtilSkipParallelPrimitives] = "1";
  lbvhXABComputeUtilId = ComputeUtil::create(compute, lbvhXABSetting, &utilInclude);

  map<ComputeUtilKey, string> lbvhSortSetting;
  lbvhSortSetting[ComputeUtilStructType] = "uint";
  lbvhSortSetting[ComputeUtilStructTypeIntegral] = "1";
  lbvhSortComputeUtilId = ComputeUtil::create(compute, lbvhSortSetting, &utilInclude);
}

void LBVHSolver::build(uint instanceNodeCount, ComputeMemory* globalOffsets)
{
  particleLeafData.resize(instanceNodeCount, false);
  particleLeafDataSorted.resize(instanceNodeCount, false);
  particleBoundingBoxes.resize(instanceNodeCount, false);
  visitedInternalNodes.resize(instanceNodeCount, false);
  leafParentNodeIndices.resize(instanceNodeCount, false);
  treeInternalNodes.resize(instanceNodeCount - 1, false);
  treeInternalNodeBoundingBoxes.resize(instanceNodeCount - 1, false);

  size_t workgroupSize[3], workgroupCount[3];
  compute->configureSize(workgroupSize, workgroupCount, instanceNodeCount);

  {
    // compute axis aligned bounding boxes for particles
    ComputeMemory* buffers[] = {
      particleBoundingBoxes.device(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_PREDICTED)->get(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_SHARED)->get(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_AUX)->get(),
      allocator->getHeap(COMPUTE_HEAP_PARTITIONS)->get(),
      allocator->getHeap(COMPUTE_HEAP_SECTIONS)->get(),
      globalOffsets
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    kernels[LBVH_COLLISION_SOLVER_CREATE_BOUNDING_BOX].setArgs(buffers, bufferCount);
    kernels[LBVH_COLLISION_SOLVER_CREATE_BOUNDING_BOX].setArg<uint>(&instanceNodeCount, bufferCount);

    compute->execute(kernels[LBVH_COLLISION_SOLVER_CREATE_BOUNDING_BOX], workgroupSize, workgroupCount);
  }

#ifdef DEBUG_LBVH_SOLVER
  particleBoundingBoxes.syncHost();
  compute->sync();
#endif

  // find bounding box for the simulation space
  ComputeUtil::get(lbvhXABComputeUtilId)->sum1D(compute, systemBoundingBox.device(), particleBoundingBoxes.device(), instanceNodeCount);

#ifdef DEBUG_LBVH_SOLVER
  /*
  systemBoundingBox.host()->push_back(XAB());
  systemBoundingBox.host()->at(0).min[0] = -10.f;
  systemBoundingBox.host()->at(0).min[1] = -10.f;
  systemBoundingBox.host()->at(0).min[2] = -10.f;
  systemBoundingBox.host()->at(0).max[0] = 10.f;
  systemBoundingBox.host()->at(0).max[1] = 10.f;
  systemBoundingBox.host()->at(0).max[2] = 10.f;
  systemBoundingBox.syncDevice();
  */
  systemBoundingBox.syncHost();
  compute->sync();
#endif

  {
    // assign morton code to the particle bounding boxes
    ComputeMemory* buffers[] = {
      particleLeafData.device(),
      systemBoundingBox.device(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_PREDICTED)->get(),
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    kernels[LBVH_COLLISION_SOLVER_ASSIGN_MORTON_CODE].setArgs(buffers, bufferCount);
    kernels[LBVH_COLLISION_SOLVER_ASSIGN_MORTON_CODE].setArg<uint>(&instanceNodeCount, bufferCount);

    compute->execute(kernels[LBVH_COLLISION_SOLVER_ASSIGN_MORTON_CODE], workgroupSize, workgroupCount);
  }

#ifdef DEBUG_LBVH_SOLVER
  particleLeafData.syncHost();
  compute->sync();
#endif

  ComputeUtil::get(lbvhSortComputeUtilId)->radixSort32Bit(compute, particleLeafDataSorted.device(), particleLeafData.device(), instanceNodeCount);

#ifdef DEBUG_LBVH_SOLVER
  particleLeafDataSorted.syncHost();
  compute->sync();
#endif

  {
    // create binary radix tree
    ComputeMemory* buffers[] = {
      treeInternalNodes.device(),
      visitedInternalNodes.device(),
      leafParentNodeIndices.device(),
      particleLeafDataSorted.device()
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    kernels[LBVH_COLLISION_SOLVER_CREATE_BINARY_TREE].setArgs(buffers, bufferCount);
    kernels[LBVH_COLLISION_SOLVER_CREATE_BINARY_TREE].setArg<uint>(&instanceNodeCount, bufferCount);

    compute->execute(kernels[LBVH_COLLISION_SOLVER_CREATE_BINARY_TREE], workgroupSize, workgroupCount);
  }

#ifdef DEBUG_LBVH_SOLVER
  treeInternalNodes.syncHost();
  leafParentNodeIndices.syncHost();
  visitedInternalNodes.syncHost();
  compute->sync();
#endif

  {
    // calculate bounding boxes for the tree
    ComputeMemory* buffers[] = {
      treeInternalNodeBoundingBoxes.device(),
      visitedInternalNodes.device(),
      treeInternalNodes.device(),
      leafParentNodeIndices.device(),
      particleBoundingBoxes.device()
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    kernels[LBVH_COLLISION_SOLVER_TREE_BOUNDING_BOX].setArgs(buffers, bufferCount);
    kernels[LBVH_COLLISION_SOLVER_TREE_BOUNDING_BOX].setArg<uint>(&instanceNodeCount, bufferCount);

    compute->execute(kernels[LBVH_COLLISION_SOLVER_TREE_BOUNDING_BOX], workgroupSize, workgroupCount);
  }

#ifdef DEBUG_LBVH_SOLVER
  treeInternalNodeBoundingBoxes.syncHost();
  visitedInternalNodes.syncHost();
  compute->sync();
#endif
  }

void LBVHSolver::solve(uint instanceNodeCount, ComputeMemory* globalOffsets)
{
  build(instanceNodeCount, globalOffsets);

  /*
  particleData.resize(instanceNodeCount, false);
  splitIndex.resize(instanceNodeCount, false);
  treeNodes.resize(instanceNodeCount, false);

  size_t workgroupSize[3], workgroupCount[3];
  compute->configureSize(workgroupSize, workgroupCount, instanceNodeCount);

  kernels[COLLISION_LBVH_SOLVER_MORTON_CODE].setArg(particleData.device(), 0);
  kernels[COLLISION_LBVH_SOLVER_MORTON_CODE].setArg(allocator->getHeap(COMPUTE_HEAP_PARTICLE_PREDICTED)->get(), 1);
  kernels[COLLISION_LBVH_SOLVER_MORTON_CODE].setArg<uint>(&instanceNodeCount, 2);

  compute->execute(kernels[COLLISION_LBVH_SOLVER_MORTON_CODE], workgroupSize, workgroupCount);

#ifdef DEBUG_LBVH_SOLVER
  particleData.syncHost();
  compute->sync();
#endif

  kernels[COLLISION_SOLVER_KERNEL_BOUNDARY].setArg(particleData.device(), 0);
  kernels[COLLISION_SOLVER_KERNEL_BOUNDARY].setArg(splitIndex.device(), 1);
  kernels[COLLISION_LBVH_SOLVER_MORTON_CODE].setArg<uint>(&instanceNodeCount, 2);

  compute->execute(kernels[COLLISION_LBVH_SPLIT_KERNEL], workgroupSize, workgroupCount);
  */
  /*size_t workgroupSize[3], workgroupCount[3];
  compute->configureSize(workgroupSize, workgroupCount, instanceNodeCount);

  ComputeMemory* buffers[] = {
  allocator->getHeap(COMPUTE_HEAP_PARTICLE)->get(),
  allocator->getHeap(COMPUTE_HEAP_PARTICLE_PREDICTED)->get(),
  allocator->getHeap(COMPUTE_HEAP_PARTICLE_IDENTITY)->get(),
  allocator->getHeap(COMPUTE_HEAP_PARTICLE_SHARED)->get(),
  allocator->getHeap(COMPUTE_HEAP_PARTICLE_AUX)->get(),
  allocator->getHeap(COMPUTE_HEAP_PARTITIONS)->get(),
  allocator->getHeap(COMPUTE_HEAP_SECTIONS)->get(),
  globalOffsets
  };
  uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
  kernels[COLLISION_SOLVER_KERNEL_BOUNDARY].setArgs(buffers, bufferCount);
  kernels[COLLISION_SOLVER_KERNEL_BOUNDARY].setArg<uint>(&instanceNodeCount, bufferCount);

  compute->execute(kernels[COLLISION_SOLVER_KERNEL_BOUNDARY], workgroupSize, workgroupCount);*/
}