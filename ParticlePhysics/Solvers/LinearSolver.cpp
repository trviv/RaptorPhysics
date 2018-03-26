#include "LinearSolver.h"

template<class IndexType, class CoefficientType, class VariableType>
LinearSolver<IndexType, CoefficientType, VariableType>::LinearSolver(ComputeInterface* compute, SharedAllocator* allocator) :
Solver<IndexType, CoefficientType, VariableType>(compute, allocator, SOLVER_EQUATION)
{
  iterations = 24;
}

template<class IndexType, class CoefficientType, class VariableType>
void LinearSolver<IndexType, CoefficientType, VariableType>::commit(const SectionData& sectionData)
{
  Solver::commit(sectionData);

  /*flatArray<CoefficientType>(*constrainCoefficients.host(), rawConstrainCoefficients);

  SectionData updateInfo;

  updateInfo.offsets[SECTION_DATA_NODE] = nodes();
  updateInfo.counts[SECTION_DATA_NODE] = constrainConstants.host()->size() - nodes();
  updateInfo.offsets[SECTION_DATA_CONNECTION] = connectionCount();
  updateInfo.counts[SECTION_DATA_CONNECTION] = constrainCoefficients.host()->size() - connectionCount();
  updateInfo.offsets[SECTION_DATA_COMMON_NODE] = connectionCount();
  updateInfo.counts[SECTION_DATA_COMMON_NODE] = constrainConstants.host()->size() - commonNodeCount();

  updates.push_back(updateInfo);*/

  //nodeOffset += updateInfo.counts[SECTION_DATA_NODE];
  //connectionOffset += updateInfo.counts[SECTION_DATA_CONNECTION];
}

template<class IndexType, class CoefficientType, class VariableType>
void LinearSolver<IndexType, CoefficientType, VariableType>::solve()
{
  if (updates.size()) // update arrays
  {
    update();
  }

  uint zero = 0;
  compute->setBuffer(constrainVariableAux[1].device(), 0, constrainVariableAux[1].size()*sizeof(VariableType), &zero, sizeof(uint));

  size_t workgroupSize[3], workgroupCount[3];
  uint count = nodes();
  compute->configureSize(workgroupSize, workgroupCount, count);
  for (uint i = 0; i < iterations; i++)
  {
    ComputeMemory* buffers[] = {
      constrainVariableAux[i & 1].device(),
      constrainHeaders.device(),
      constrainIndices.device(),
      constrainCoefficients.device(),
      constrainVariableAux[(i + 1) & 1].device(),
      constrainConstants.device()
    };
    kernels[0].setArgs(buffers, 6);
    kernels[0].setArg<uint>(&count, 6);
    compute->execute(kernels[0], workgroupSize, workgroupCount);
  }

  VariableType* values = new VariableType[nodes()];
  compute->copyToHost(constrainVariableAux[(iterations - 1) & 1].device(), 0, nodes()*sizeof(VariableType), values, true);

  for (uint i = 0; i < nodes(); i++)
  {
    std::cout << values[i] << "\n";
  }

  delete[] values;
}

template<class IndexType, class CoefficientType, class VariableType>
void LinearSolver<IndexType, CoefficientType, VariableType>::update()
{
  Solver::update();
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