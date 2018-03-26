#ifndef SOLVER_H
#define SOLVER_H

#include "SolverData.h"

#define DEBUG_SOLVERS

enum SolverType
{
  SOLVER_NULL = 0,
  SOLVER_CLOTH = 1,
  SOLVER_RIGID_BODY = 2,
  SOLVER_EQUATION = 4,

  SOLVER_MAX = 3
};

template<class IndexType, class CoefficientType, class VariableType> class Solver :
public SolverData<IndexType, CoefficientType, VariableType>, public ShaderEntity
{
protected:

  ComputeInterface*   compute;
  SharedAllocator*    allocator;
  SolverType          type;
  uint                iterations;

  friend class PhysicsSystem;

  template<class BaseType> void flatArray(vector<BaseType>& out, const vector<vector<BaseType>>& in)
  {
    out.clear();
    for (uint i = 0; i < in.size(); i++)
    {
      for (uint j = 0; j < in[i].size(); j++)
      {
        out.push_back(in[i][j]);
      }
    }
  }

  virtual void update();

public:

  Solver(ComputeInterface* compute, SharedAllocator* allocator, SolverType type);

  ~Solver();

  virtual void commit(const SectionData& sectionData);

  virtual void create(ComputeInterface* compute) = 0;

  virtual void solve() = 0;

  /*@function Get an available unique entity id.*/
  uint newEntityId();

  uint uniqueEntityCount()const;

  uint totalEntityCount()const;
};

#endif