#include "SolverData.h"

template<class IndexType, class CoefficientType, class VariableType>
SolverData<IndexType, CoefficientType, VariableType>::SolverData()
{
  updates.clear();
  rawConstrainConnections.clear();
  rawConstrainCoefficients.clear();
}

template<class IndexType, class CoefficientType, class VariableType>
uint SolverData<IndexType, CoefficientType, VariableType>::nodes()const
{
  uint ret = 0;

  if (entityLocations.host() && entityLocations.host()->size())
  {
    ret = entityLocations.host()->back().node.end();
  }
  if (updates.size() > 0)
  {
    ret = updates.back().node.end();
  }

  return ret;
}

template<class IndexType, class CoefficientType, class VariableType>
uint SolverData<IndexType, CoefficientType, VariableType>::connectionCount()const
{
  uint ret = 0;

  if (entityLocations.host() && entityLocations.host()->size())
  {
    ret = entityLocations.host()->back().connection.end();
  }
  if (updates.size() > 0)
  {
    ret = updates.back().connection.end();
  }

  return ret;
}

template<class IndexType, class CoefficientType, class VariableType>
const PartitionInfo SolverData<IndexType, CoefficientType, VariableType>::lastPartition()const
{
  return partitions.host()->size() ? partitions.host()->back() : PartitionInfo(0);
}

template<class IndexType, class CoefficientType, class VariableType>
void SolverData<IndexType, CoefficientType, VariableType>::addCoefficient(IndexType index, CoefficientType coefficient)
{
  // add coeficient to first element
  expand<SingleCoefficient>(index + nodes(), rawConstrainCoefficients);
  rawConstrainCoefficients[index + nodes()].push_back(coefficient);
}

template<class IndexType, class CoefficientType, class VariableType>
void SolverData<IndexType, CoefficientType, VariableType>::addConstrain(IndexType index, IndexType connection)
{
  // add contrain to first element
  expand<SingleConstrain>(index + nodes(), rawConstrainConnections);
  rawConstrainConnections[index + nodes()].push_back(connection + nodes());
}

template<class IndexType, class CoefficientType, class VariableType>
void SolverData<IndexType, CoefficientType, VariableType>::addConnection(IndexType index, IndexType connection, CoefficientType coefficient)
{
  addConstrain(index, connection);
  addCoefficient(index, coefficient);
}

template<class IndexType, class CoefficientType, class VariableType>
void SolverData<IndexType, CoefficientType, VariableType>::setConstant(IndexType index, VariableType value)
{
  //add value for the element
  expand<VariableType>(index + nodes(), *constrainConstants.host());
  (*constrainConstants.host())[index + nodes()] = value;
}

#define classPrefix(x, y, z) template void SolverData<x, y, z>

#define declareFunctions(x, y, z) \
  template SolverData<x, y, z>::SolverData(); \
  template uint SolverData<x, y, z>::nodes()const; \
  template uint SolverData<x, y, z>::connectionCount()const; \
  template const PartitionInfo SolverData<x, y, z>::lastPartition()const; \
  classPrefix(x, y, z)::addConstrain(x index, x connection); \
  classPrefix(x, y, z)::addCoefficient(x index, y coefficient); \
  classPrefix(x, y, z)::addConnection(x index, x connection, y coefficient); \
  classPrefix(x, y, z)::setConstant(x index, z value);

declareFunctions(ushort, real, real)
declareFunctions(uint, real, real)
declareFunctions(ushort, real, Real3)
declareFunctions(uint, real, Real3)
