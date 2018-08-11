#include "LBVHSolver.h"

#define DEBUG_LBVH_SOLVER

#define COLLISION_SOLVER_KERNEL_BOUNDARY  0
#define COLLISION_LBVH_SOLVER_MORTON_CODE 1
#define COLLISION_LBVH_SPLIT_KERNEL       2

uint mortonSortUtilId = -1;

DeviceArray <uint> splitIndex;
DeviceArray <BVHNodeInfo> treeNodes;

LBVHSolver::~LBVHSolver()
{
}

void LBVHSolver::init(ComputeInterface* compute, SharedAllocator* allocator)
{
  CollisionSolver::init(compute, allocator);
  registerShader(compute, "LBVHSolver.shader", NULL, NULL);
  kernels.push_back(programs[1].createKernel("findSplitKernel"));

#ifdef DEBUG_LBVH_SOLVER
  particleData.create(compute, NULL, true);
  splitIndex.create(compute, NULL, true);
  treeNodes.create(compute, NULL, true);
#else
  particleData.create(compute, NULL, false);
  splitIndex.create(compute, NULL, false);
  treeNodes.create(compute, NULL, false);
#endif

  map<ComputeUtilKey, string> mortonSortSetting;
  mortonSortSetting[ComputeUtilStructType] = "uint";
  mortonSortSetting[ComputeUtilStructTypeIntegral] = "1";
  mortonSortUtilId = ComputeUtil::create(compute, mortonSortSetting, NULL);
}

void LBVHSolver::solve(uint instanceNodeCount, ComputeMemory* globalOffsets)
{
  //build(instanceNodeCount);

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