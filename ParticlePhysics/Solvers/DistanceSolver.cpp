/*
 * Copyright (c) Vivek Trivedi, 2025
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#include "DistanceSolver.h"

#define DISTANCE_SOLVER_KERNEL_SPRING 0

//#define DEBUG_DISTANCE_SOLVER

DistanceSolver::DistanceSolver(ComputeInterface* compute, SharedAllocator* allocator)
  : Solver(compute, allocator), LinearSolver(compute, allocator)
{
  type = SOLVER_CLOTH;
  iterations = 16;
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
  }
}

void DistanceSolver::solve(float timeStep)
{
  uint count = lastPartition().end();

  if (!count) return;

  size_t workgroupSize[3], workgroupCount[3];
  compute->configureSize(workgroupSize, workgroupCount, count);

  particlesTemp[0].resize(count, false);

  for (uint i = 0; i < iterations; i++)
  {
    ComputeMemory* newPosition = ((i & 1) == 1) ? particlesPredicted.device() : particlesTemp[0].device();
    ComputeMemory* oldPosition = ((i & 1) == 0) ? particlesPredicted.device() : particlesTemp[0].device();

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
    compute->execute(kernels[DISTANCE_SOLVER_KERNEL_SPRING], workgroupSize, workgroupCount);

#if defined(DEBUG_DISTANCE_SOLVER) && defined(DEBUG_SOLVERS)
    particlesTemp[0].syncHost();
    compute->sync();
#endif

  }

  if ((iterations & 1) == 1)
  {
    ComputeUtil::get(ComputeUtil::getUInt4Util(compute))->copyBuffer(compute, particlesTemp[0].device(), particlesPredicted.device(), 0, 0, count * sizeof(ParticleStruct));
  }
#if defined(DEBUG_DISTANCE_SOLVER) && defined(DEBUG_SOLVERS)
  compute->sync();
#endif
}

void DistanceSolver::update()
{
  if (!updates.size()) return;

  LinearSolver::update();
}
