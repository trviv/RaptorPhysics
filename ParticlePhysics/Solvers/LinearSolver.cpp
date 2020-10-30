#include "LinearSolver.h"

template<class IndexType, class CoefficientType, class VariableType>
LinearSolver<IndexType, CoefficientType, VariableType>::LinearSolver(ComputeInterface* compute, SharedAllocator* allocator) :
  Solver(compute, allocator), EntitySolver<IndexType, CoefficientType, VariableType>(compute, allocator, SOLVER_EQUATION)
{
  this->iterations = 24;
}

template<class IndexType, class CoefficientType, class VariableType>
void LinearSolver<IndexType, CoefficientType, VariableType>::solve()
{
  uint zero = 0;
  ComputeUtil::get(ComputeUtil::getUIntUtil(this->compute))->clearBuffer(this->compute, this->constrainVariableAux[1].device(), this->constrainVariableAux[1].size()*sizeof(VariableType), *((uint*)&zero));

  size_t workgroupSize[3], workgroupCount[3];
  uint count = this->nodes();
  this->compute->configureSize(workgroupSize, workgroupCount, count);
  for (uint i = 0; i < this->iterations; i++)
  {
    ComputeMemory* buffers[] = {
      this->constrainVariableAux[i & 1].device(),
      this->constrainHeaders.device(),
      this->constrainIndices.device(),
      this->constrainCoefficients.device(),
      this->constrainVariableAux[(i + 1) & 1].device(),
      this->constrainConstants.device()
    };
    this->kernels[0].setArgs(buffers, 6);
    this->kernels[0].template setArg<uint>(&count, 6);
    this->compute->execute(this->kernels[0], workgroupSize, workgroupCount);
  }

  VariableType* values = new VariableType[this->nodes()];
  this->compute->copyToHost(this->constrainVariableAux[(this->iterations - 1) & 1].device(), 0, this->nodes()*sizeof(VariableType), values, true);

  for (uint i = 0; i < this->nodes(); i++)
  {
    std::cout << values[i] << "\n";
  }

  delete[] values;
}

template<class IndexType, class CoefficientType, class VariableType>
void LinearSolver<IndexType, CoefficientType, VariableType>::update()
{
  EntitySolver<IndexType, CoefficientType, VariableType>::update();
}

#define classPrefix(x, y, z) template void LinearSolver<x, y, z>

#define declareFunctions(x, y, z) \
  template LinearSolver<x, y, z>::LinearSolver(ComputeInterface* compute, SharedAllocator* allocator); \
  classPrefix(x, y, z)::update(); \
  classPrefix(x, y, z)::solve();

template<>
void LinearSolver<uint, real, real>::create(ComputeInterface* compute)
{
  if (programs.empty())
  {
    vector<string> newType = { "uint", "float", "float" };
    vector<string> oldType = { "IndexType", "CoefficientType", "VariableType" };
    registerShader(compute, "LinearSolver.shader", &oldType, &newType);
    kernels.push_back(programs[0].createKernel("linearSolver"));
  }
}

template<>
void LinearSolver<ushort, real, real>::create(ComputeInterface* compute)
{
  if (programs.empty())
  {
    vector<string> newType = { "ushort", "float", "float" };
    vector<string> oldType = { "IndexType", "CoefficientType", "VariableType" };
    registerShader(compute, "LinearSolver.shader", &oldType, &newType);
    kernels.push_back(programs[0].createKernel("linearSolver"));
  }
}

template<>
void LinearSolver<uint, real, Real3>::create(ComputeInterface* compute)
{
  if (programs.empty())
  {
    vector<string> newType = { "uint", "float", "float3" };
    vector<string> oldType = { "IndexType", "CoefficientType", "VariableType" };
    registerShader(compute, "LinearSolver.shader", &oldType, &newType);
    kernels.push_back(programs[0].createKernel("linearSolver"));
  }
}

declareFunctions(ushort, real, real)
declareFunctions(uint, real, Real3)
