#include "SolverData.h"

template<class IndexType, class CoefficientType, class VariableType>
SolverData<IndexType, CoefficientType, VariableType>::SolverData()
{
}

template<class IndexType, class CoefficientType, class VariableType>
uint SolverData<IndexType, CoefficientType, VariableType>::nodes()const
{
  uint ret = 0;

  if (deviceSections.host() && deviceSections.host()->size())
  {
    ret = deviceSections.host()->back().offsets[SECTION_DATA_NODE] + deviceSections.host()->back().counts[SECTION_DATA_NODE];
  }
  if (updates.size() > 0)
  {
    ret = updates.back().offsets[SECTION_DATA_NODE] + updates.back().counts[SECTION_DATA_NODE];
  }

  return ret;
}

template<class IndexType, class CoefficientType, class VariableType>
uint SolverData<IndexType, CoefficientType, VariableType>::commonNodeCount()const
{
  uint ret = 0;

  if (deviceSections.host() && deviceSections.host()->size())
  {
    ret = deviceSections.host()->back().offsets[SECTION_DATA_COMMON_NODE] + deviceSections.host()->back().counts[SECTION_DATA_COMMON_NODE];
  }
  if (updates.size() > 0)
  {
    ret = updates.back().offsets[SECTION_DATA_COMMON_NODE] + updates.back().counts[SECTION_DATA_COMMON_NODE];
  }

  return ret;
}

template<class IndexType, class CoefficientType, class VariableType>
uint SolverData<IndexType, CoefficientType, VariableType>::connectionCount()const
{
  uint ret = 0;

  if (deviceSections.host() && deviceSections.host()->size())
  {
    ret = deviceSections.host()->back().offsets[SECTION_DATA_CONNECTION] + deviceSections.host()->back().counts[SECTION_DATA_CONNECTION];
  }
  if (updates.size() > 0)
  {
    ret = updates.back().offsets[SECTION_DATA_CONNECTION] + updates.back().counts[SECTION_DATA_CONNECTION];
  }

  return ret;
}

template<class IndexType, class CoefficientType, class VariableType>
void SolverData<IndexType, CoefficientType, VariableType>::addCoefficient(IndexType index, CoefficientType coefficient)
{
  // add coeficient to first element
  //expand<SingleCoefficient>(index + nodes(), constrainCoefficients);
  //constrainCoefficients[index + nodes()].push_back(coefficient);
  expand<SingleCoefficient>(index + nodes(), rawConstrainCoefficients);
  rawConstrainCoefficients[index + nodes()].push_back(coefficient);
}

template<class IndexType, class CoefficientType, class VariableType>
void SolverData<IndexType, CoefficientType, VariableType>::addConstrain(IndexType index, IndexType connection)
{
  // add contrain to first element
  //expand<SingleConstrain>(index + nodes(), constrainConnections);
  //constrainConnections[index + nodes()].push_back(connection + nodes());
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
  //expand<VariableType>(index + nodes(), *deviceConstrainConstants.host());
  //(*deviceConstrainConstants.host())[index + nodes()] = value;
  expand<VariableType>(index + nodes(), *constrainConstants.host());
  (*constrainConstants.host())[index + nodes()] = value;
}

#define classPrefix(x, y, z) template void SolverData<x, y, z>

#define declareFunctions(x, y, z) \
  template SolverData<x, y, z>::SolverData(); \
  template uint SolverData<x, y, z>::nodes()const; \
  template uint SolverData<x, y, z>::connectionCount()const; \
  template uint SolverData<x, y, z>::commonNodeCount()const; \
  classPrefix(x, y, z)::addConstrain(x index, x connection); \
  classPrefix(x, y, z)::addCoefficient(x index, y coefficient); \
  classPrefix(x, y, z)::addConnection(x index, x connection, y coefficient); \
  classPrefix(x, y, z)::setConstant(x index, z value);

declareFunctions(ushort, real, real)
declareFunctions(uint, real, real)
declareFunctions(ushort, real, Real3)
declareFunctions(uint, real, Real3)