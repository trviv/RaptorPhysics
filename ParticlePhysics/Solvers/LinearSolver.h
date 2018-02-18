#ifndef LINEAR_SOLVER_H
#define LINEAR_SOLVER_H

#include "Solver.h"

template<class IndexType, class CoefficientType, class VariableType> class LinearSolver :
public Solver<IndexType, CoefficientType, VariableType>
{
protected:

  void update();

public:

  LinearSolver(ComputeInterface* compute, SharedAllocator* allocator);

  void commit();

  void create(ComputeInterface* compute);

  void solve();
};

#endif