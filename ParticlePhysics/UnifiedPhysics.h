/*
 * Copyright (c) Vivek Trivedi, 2025
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#ifndef UNIFIED_PHYSICS_H
#define UNIFIED_PHYSICS_H

#include "Solvers/Constrain.h"
#include "Solvers/SolverData.h"
#include "Solvers/Solver.h"
#include "Solvers/LinearSolver.h"
#include "Solvers/DistanceSolver.h"
#include "Solvers/RigidSolver.h"
#include "Solvers/FluidSolver.h"
#include "Solvers/FluidSolverPBF.h"
#include "Solvers/FluidSolverPCISPH.h"
#include "Solvers/SharedAllocator.h"

#include "Solvers/Collision/CollisionSolver.h"
#include "Solvers/Collision/LBVHSolver.h"

#include "Common/ConstrainStruct.h"
#include "Common/ParticleStruct.h"

#include "Entities/PhysicsEntity.h"
#include "Entities/Cloth.h"
#include "Entities/RigidBody.h"
#include "Entities/Fluid.h"
#include "Entities/PhysicsSystem.h"

#endif
