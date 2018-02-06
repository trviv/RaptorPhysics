#include "PhysicsEntity.h"

PhysicsEntity::PhysicsEntity()
{
  constrainConstants.create(NULL, NULL, true);
  particleSharedData.create(NULL, NULL, true);
  particleAuxData.create(NULL, NULL, true);

  instanceCount = 1;
  solver = 0;
}