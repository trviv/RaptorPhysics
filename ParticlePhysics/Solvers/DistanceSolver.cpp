#include "DistanceSolver.h"

#define DISTANCE_SOLVER_KERNEL_SPRING 0
#define DISTANCE_SOLVER_KERNEL_SET_DELTA_POSITION 1

//#define DEBUG_DISTANCE_SOLVER

DistanceSolver::DistanceSolver(ComputeInterface* compute, SharedAllocator* allocator)
  : LinearSolver(compute, allocator)
{
  type = SOLVER_CLOTH;
  iterations = 2;
  create(compute);
}

void DistanceSolver::addDistance(IndexType index, IndexType connectionIndex, CoefficientType distance)
{
  addConnection(index, connectionIndex, distance);
}

void DistanceSolver::create(ComputeInterface* compute)
{
  if (programs.empty())
  {
    vector<string> newType = { "uint", "float", "float3" };
    vector<string> oldType = { "IndexType", "CoefficientType", "VariableType" };
    registerShader(compute, "DistanceSolver.shader", &oldType, &newType);
    kernels.push_back(programs[0].createKernel("distanceSolverSpring"));
    kernels.push_back(programs[0].createKernel("setDeltaPosition"));
  }
}

void DistanceSolver::solve()
{
  uint count = lastPartition().end();

  if (!count) return;

  if (updates.size())
  {
    update();
  }

  size_t workgroupSize[3], workgroupCount[3];
  compute->configureSize(workgroupSize, workgroupCount, count);

  particlesTemp[0].resize(count, false);
  particlesTemp[1].resize(count, false);

  for (uint i = 0; i < iterations; i++)
  {
    ComputeMemory* newPosition = particlesTemp[i & 1].device();
    ComputeMemory* oldPosition = (i == 0) ? particlesPredicted.device() : particlesTemp[(i + 1) & 1].device();

    if (i == 0)
    {
      // set everything in the first iteration
      ComputeMemory* buffers[] = {
        newPosition,
        oldPosition,
        constrainHeaders.device(),
        constrainIndices.device(),
        constrainCoefficients.device(),
        partitions.device(),
        entityLocations.device()
      };
      uint bufferOffset = sizeof(buffers) / sizeof(ComputeMemory*);
      kernels[DISTANCE_SOLVER_KERNEL_SPRING].setArgs(buffers, bufferOffset);
      kernels[DISTANCE_SOLVER_KERNEL_SPRING].setArg<uint>(&count, bufferOffset);
    }
    else
    {
      // set only the changed data for second plus iteration
      ComputeMemory* buffers[] = {
        newPosition,
        oldPosition
      };
      uint bufferOffset = sizeof(buffers) / sizeof(ComputeMemory*);
      kernels[DISTANCE_SOLVER_KERNEL_SPRING].setArgs(buffers, bufferOffset);
    }
    compute->execute(kernels[DISTANCE_SOLVER_KERNEL_SPRING], workgroupSize, workgroupCount);

#if defined(DEBUG_DISTANCE_SOLVER) && defined(DEBUG_SOLVERS)
    particlesTemp[0].syncHost();
    particlesTemp[1].syncHost();
    compute->sync();
#endif

  }

  { // calculate position deltas
    ComputeMemory* buffers[] = {
      particleDeltas.device(),
      particlesPredicted.device(),
      particlesTemp[(iterations - 1) & 1].device()
    };
    uint bufferOffset = sizeof(buffers) / sizeof(ComputeMemory*);
    kernels[DISTANCE_SOLVER_KERNEL_SET_DELTA_POSITION].setArgs(buffers, bufferOffset);
    kernels[DISTANCE_SOLVER_KERNEL_SET_DELTA_POSITION].setArg<uint>(&count, bufferOffset);
    compute->execute(kernels[DISTANCE_SOLVER_KERNEL_SET_DELTA_POSITION], workgroupSize, workgroupCount);
  }
#if defined(DEBUG_DISTANCE_SOLVER) && defined(DEBUG_SOLVERS)
  compute->sync();
#endif
}

void DistanceSolver::update()
{
  LinearSolver::update();
}
