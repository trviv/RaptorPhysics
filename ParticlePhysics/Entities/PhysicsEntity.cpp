#include "PhysicsEntity.h"

PhysicsEntity::PhysicsEntity()
{
  constrainConstants.create(NULL, NULL, true);
  entitySharedData.create(NULL, NULL, true);
  particleAuxData.create(NULL, NULL, true);
  particleRigidData.create(NULL, NULL, true);
  particleCollisionData.create(NULL, NULL, true);

  solver = SOLVER_NULL;

  ParticleSharedData sharedData;
  sharedData.velocityDamping = .98f;
  sharedData.setInvMassIsShared(true);
  sharedData.sharedInvMass = 0.f;
  sharedData.setRadiusIsShared(true);
  sharedData.sharedRadius = 0.f;
  sharedData.setCollisionDataIsShared(true);
  sharedData.sharedCollisionData.initialSdfGradient = float3(0, 0, 0);
  sharedData.sharedCollisionData.sdfMagnitude = 0.f;
  sharedData.sharedCollisionData.transformedSdfGradient = float3(0, 0, 0);
  entitySharedData.host()->reserve(1);
  entitySharedData.host()->push_back(sharedData);
}