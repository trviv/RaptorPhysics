#include "CollisionSolver.h"

#define COLLISION_SOLVER_KERNEL_BOUNDARY  0
#define COLLISION_LBVH_SOLVER_MORTON_CODE 1

CollisionSolver::~CollisionSolver()
{
}

void CollisionSolver::init(ComputeInterface* compute, SharedAllocator* allocator)
{
  this->compute = compute;
  this->allocator = allocator;

  includeFiles.push_back("ComputeHeader.shader");
  includeFiles.push_back("ComputeShared.h");
  includeFiles.push_back("ParticleStruct.h");
  includeFiles.push_back("CollisionSolverShared.h");

  registerShader(compute, "CollisionSolver.shader", NULL, NULL);
  kernels.push_back(programs[0].createKernel("boundaryCollisionKernel"));
  kernels.push_back(programs[0].createKernel("assignMortonCodeKernel"));
}

void CollisionSolver::build(uint instanceNodeCount)
{
  /*
  size_t workgroupSize[3], workgroupCount[3];
  compute->configureSize(workgroupSize, workgroupCount, instanceNodeCount);

  kernels[COLLISION_LBVH_SOLVER_MORTON_CODE].setArg(allocator->getHeap(COMPUTE_HEAP_PARTICLE), 0);
  kernels[COLLISION_LBVH_SOLVER_MORTON_CODE].setArg<uint>(&instanceNodeCount, 1);
  compute->execute(kernels[COLLISION_LBVH_SOLVER_MORTON_CODE], workgroupSize, workgroupCount);
  */
}

void CollisionSolver::solve(uint instanceNodeCount, ComputeMemory* globalOffsets)
{
  size_t workgroupSize[3], workgroupCount[3];
  compute->configureSize(workgroupSize, workgroupCount, instanceNodeCount);

  build(instanceNodeCount);

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

  compute->execute(kernels[COLLISION_SOLVER_KERNEL_BOUNDARY], workgroupSize, workgroupCount);
}