#include "DistanceSolver.h"

static const int DISTANCE_SOLVER_SPRING_KERNEL = 0;
static const int SET_DELTA_POSITION_KERNEL = 1;

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
    vector<string> oldType = { "uint", "float", "float3" };
    vector<string> newType = { "IndexType", "CoefficientType", "VariableType" };
    registerShader(compute, "DistanceSolver.shader", &oldType, &newType);
    kernels.push_back(programs[0].createKernel("distanceSolverSpring"));
    kernels.push_back(programs[0].createKernel("setDeltaPosition"));
  }
}

void DistanceSolver::solve()
{
  if (updates.size())
  {
    update();
  }

  size_t workgroupSize[3], workgroupCount[3];
  uint count = nodes();
  compute->configureSize(workgroupSize, workgroupCount, count);

  particleDiff.resize(count, false);
  particlesTemp[0].resize(count, false);
  particlesTemp[1].resize(count, false);

  for (uint i = 0; i < iterations; i++)
  {
    ComputeMemory* newPosition = particlesTemp[i & 1].device();
    ComputeMemory* oldPosition = (i == 0) ? particles.device() : particlesTemp[(i + 1) & 1].device();

    ComputeMemory* buffers[] = {
      newPosition,
      oldPosition,
      constrainHeaders.device(),
      constrainIndices.device(),
      constrainCoefficients.device()
    };
    kernels[DISTANCE_SOLVER_SPRING_KERNEL].setArgs(buffers, 5);
    kernels[DISTANCE_SOLVER_SPRING_KERNEL].setArg<uint>(&count, 5);
    compute->execute(kernels[DISTANCE_SOLVER_SPRING_KERNEL], workgroupSize, workgroupCount);

#if defined(DEBUG_DISTANCE_SOLVER) && defined(DEBUG_SOLVERS)
    compute->sync();
    particlesTemp[0].syncHost();
    particlesTemp[1].syncHost();
#endif

  }

  {
    ComputeMemory* buffers[] = {
      particleDeltas.device(),
      particles.device(),
      particlesTemp[(iterations - 1) & 1].device()
    };
    kernels[SET_DELTA_POSITION_KERNEL].setArgs(buffers, 3);
    kernels[SET_DELTA_POSITION_KERNEL].setArg<uint>(&count, 3);
    compute->execute(kernels[SET_DELTA_POSITION_KERNEL], workgroupSize, workgroupCount);
  }
}

void DistanceSolver::update()
{
  LinearSolver::update();

  particleSharedData.syncDevice();
  particles.syncDevice();
  particleDeltas.resize(particles.size(), false);
  particleAuxData.syncDevice();
}