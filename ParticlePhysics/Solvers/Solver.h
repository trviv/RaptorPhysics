#ifndef SOLVER_H
#define SOLVER_H

#include "SolverData.h"

#define DEBUG_SOLVERS

enum SolverType
{
  SOLVER_EQUATION = 1,
  SOLVER_CLOTH = 2,
  SOLVER_RIGID_BODY = 4,
  SOLVER_SYSTEM,

  SOLVER_MAX = 4
};

template<class IndexType, class CoefficientType, class VariableType> class Solver :
public SolverData<IndexType, CoefficientType, VariableType>, public ShaderEntity
{
protected:

  ComputeInterface*   compute;
  SharedAllocator*    allocator;
  SolverType          type;
  uint                iterations;
  vector<SectionData> updates;

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

  virtual void commit();

  virtual void create(ComputeInterface* compute) = 0;

  virtual void solve() = 0;
};

#endif