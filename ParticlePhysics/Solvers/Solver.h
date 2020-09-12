#ifndef SOLVER_H
#define SOLVER_H

#include "SolverData.h"

#define DEBUG_SOLVERS

/*!
@class Base class for any solver.
*/
class Solver : protected ShaderEntity
{
protected:

  ComputeInterface* compute;
  SharedAllocator*  allocator;
  uint              iterations;

public:

  Solver(ComputeInterface* compute, SharedAllocator* allocator);
};

/*!
@class Base class for all entity solvers.
*/
template<class IndexType, class CoefficientType, class VariableType>
class EntitySolver : virtual protected Solver, public SolverData<IndexType, CoefficientType, VariableType>
{
protected:

  SolverType type;

  friend class PhysicsSystem;

  template<class BaseType> void flatArray(vector<BaseType>& out, const vector< vector<BaseType> >& in)
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

  /*!@constructor Process all entity properties and commit to the device memory.*/
  EntitySolver(ComputeInterface* compute, SharedAllocator* allocator, SolverType type);

  /*!@destructor Process all entity properties and commit to the device memory.*/
  ~EntitySolver();

  /*!@function Process all entity properties and commit to the device memory.*/
  virtual void commit();

  /*!@function Create the entity solver.*/
  virtual void create(ComputeInterface* compute) = 0;

  /*!@function Solve the entity constrains.*/
  virtual void solve(float timeStep) = 0;

  /*!@function Apply post collision constrain constrains.*/
  virtual void postCollisionSolve(float timeStep){};

  /*!@function Get an available unique entity id.*/
  uint newEntityId();

  /*!@function Get an available unique instance id.*/
  uint newEntityInstanceId()const;
};

typedef EntitySolver<uint, real, Real3> EntitySolverType;

#endif
