#include "Light.h"

Light::Light()
  :RayTracingEntity(RayTracingEntityLight)
{
  Real3(color).setNull();
}
