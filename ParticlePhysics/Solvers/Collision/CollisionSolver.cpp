#include "CollisionSolver.h"

#define COLLISION_SOLVER_BUILD            0
#define COLLISION_SOLVER_KERNEL_BOUNDARY  1
#define COLLISION_LBVH_SOLVER_MORTON_CODE 2

#define COLLISION_DEBUG_GRID

CollisionSolver::~CollisionSolver()
{
}

void CollisionSolver::init(ComputeInterface* compute, SharedAllocator* allocator)
{
  this->compute = compute;
  this->allocator = allocator;
  gridSize = 1024;

  includeFiles.push_back("ComputeHeader.shader");
  includeFiles.push_back("ComputeShared.h");
  includeFiles.push_back("ParticleStruct.h");
  includeFiles.push_back("CollisionSolverShared.h");

  registerShader(compute, "CollisionSolver.shader", NULL, NULL);
  kernels.push_back(programs[0].createKernel("buildUniformGrid"));
  kernels.push_back(programs[0].createKernel("boundaryCollisionKernel"));
  kernels.push_back(programs[0].createKernel("assignMortonCodeKernel"));

#ifdef COLLISION_DEBUG_GRID
  gridParticleIndices.create(compute, 0, true);
  gridCellParticleCount.create(compute, 0, true);
#else
  gridParticleIndices.create(compute);
  gridCellParticleCount.create(compute);
#endif
}

void CollisionSolver::build(uint instanceNodeCount, ComputeMemory* globalOffsets)
{
  // resize index array if necessary
  if (gridParticleIndices.size() < instanceNodeCount)
  {
    gridParticleIndices.resize(instanceNodeCount, false);
  }

  // resize offset array if necessary
  if (gridCellParticleCount.size() < gridSize*gridSize*gridSize)
  {
    gridCellParticleCount.resize(gridSize*gridSize*gridSize, false);
  }

  uint zero = 0;

  // clear index offset buffer
  compute->setBuffer(gridCellParticleCount.device(), 0, gridSize*gridSize*gridSize*sizeof(uint), &zero, sizeof(uint));

  size_t workgroupSize[3], workgroupCount[3];
  compute->configureSize(workgroupSize, workgroupCount, instanceNodeCount);

  ComputeMemory* buffers[] = {
    allocator->getHeap(COMPUTE_HEAP_PARTICLE)->get(),
    allocator->getHeap(COMPUTE_HEAP_PARTICLE_PREDICTED)->get(),
    allocator->getHeap(COMPUTE_HEAP_PARTICLE_IDENTITY)->get(),
    allocator->getHeap(COMPUTE_HEAP_PARTITIONS)->get(),
    allocator->getHeap(COMPUTE_HEAP_SECTIONS)->get(),
    globalOffsets
  };
  uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
  kernels[COLLISION_SOLVER_BUILD].setArgs(buffers, bufferCount);
  kernels[COLLISION_SOLVER_BUILD].setArg<uint>(&instanceNodeCount, bufferCount);

  compute->execute(kernels[COLLISION_SOLVER_BUILD], workgroupSize, workgroupCount);

  /*size_t workgroupSize[3], workgroupCount[3];
  compute->configureSize(workgroupSize, workgroupCount, instanceNodeCount);

  kernels[COLLISION_LBVH_SOLVER_MORTON_CODE].setArg(allocator->getHeap(COMPUTE_HEAP_PARTICLE), 0);
  kernels[COLLISION_LBVH_SOLVER_MORTON_CODE].setArg<uint>(&instanceNodeCount, 1);
  compute->execute(kernels[COLLISION_LBVH_SOLVER_MORTON_CODE], workgroupSize, workgroupCount);*/
}

void CollisionSolver::solve(uint instanceNodeCount, ComputeMemory* globalOffsets)
{
  size_t workgroupSize[3], workgroupCount[3];
  compute->configureSize(workgroupSize, workgroupCount, instanceNodeCount);

  //build(instanceNodeCount, globalOffsets);

  ComputeMemory* buffers[] = {
    allocator->getHeap(COMPUTE_HEAP_PARTICLE)->get(),
    allocator->getHeap(COMPUTE_HEAP_PARTICLE_PREDICTED)->get(),
    allocator->getHeap(COMPUTE_HEAP_PARTICLE_IDENTITY)->get(),
    allocator->getHeap(COMPUTE_HEAP_PARTICLE_COLLISION)->get(),
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