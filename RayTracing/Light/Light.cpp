#include "Light.h"

Light::Light(RayTracingEntityType type)
{
  setRayTracingEntityId(this->identity, type, 0);
  ((Real3*)&color)->setNull();
  ((Real3*)&normal)->setNull();
  ((Real3*)&position)->setNull();
}

RayTracingEntity* Light::createCopy()const
{
  Light *newLight = new Light((RayTracingEntityType)getRayTracingEntityType(this->identity));
  *newLight = *this;
  return newLight;
}

RayTracingEntityId Light::getIdentity()const
{
  return identity;
}

void Light::update()
{
  const auto id = this->identity;
  transform.transformPos(position);
  this->identity = id;
}
