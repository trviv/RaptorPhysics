#include "Light.h"

Light::Light(RayTracingEntityType type)
{
  setRayTracingEntityId(this->identity, type, 0);
  ((Real3*)&up)->setNull();
  ((Real3*)&right)->setNull();
  ((Real3*)&color)->setNull();
  ((Real3*)&normal)->setNull();
  ((Real3*)&position)->setNull();
}

RayTracingEntity* Light::createCopy()const
{
  Light *newLight = new Light(getEntityType());
  *newLight = *this;
  return newLight;
}

RayTracingEntityId Light::getIdentity()const
{
  return identity;
}

void Light::setIdentity(RayTracingEntityId identity)
{
  this->identity = identity;
}

void Light::update()
{
  const auto id = this->identity;
  transform.transformPos(position);
  this->identity = id;

  if (getEntityType() == RayTracingEntityLightArea)
  {
    Real3 normal = Real3(0.f, 0.f, 1.f);
    transform.transformDir(normal);
    normal.normalize();
    this->normal = normal;

    Real3 width(1.f, 0.f, 0.f);
    transform.transformDir(width);
    this->right = width;

    Real3 height(0.f, 1.f, 0.f);
    transform.transformDir(height);
    this->up = height;
  }
}
