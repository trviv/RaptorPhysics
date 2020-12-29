#ifndef LIGHT_SHADER
#define LIGHT_SHADER

inline void sampleLight(Thread float3* position, Thread float3* color, Thread float3* direction, Thread float* maxDistance,
  Const LightStruct* light, const uint offset, const ushort bounce, const float3 hitPosition)
{
  if (getRayTracingEntityType(light->identity) == 2)
  {
    const float2 random = constructFloat2(getRandomNumber(offset, 2+bounce*4+0), getRandomNumber(offset, 2+bounce*4+1)) * 2.0f - 1.0f;
    //const half2 random = half2(getRandomNumber(offset, 2+bounce*4+0), getRandomNumber(offset, 2+bounce*4+1)) * 2.0f - 1.0f;
    *position = light->position + constructFloat3(light->right.xyz) * random.x + constructFloat3(light->up.xyz) * random.y;
    *direction = *position - hitPosition;
    *maxDistance = length(*direction);
    *direction /= *maxDistance;
    *color = light->color * max(0.f, -dot(*direction, light->normal));
  }
  else
  {
    *position = light->position;
    *color = light->color;
    *direction = *position - hitPosition;
    *maxDistance = length(*direction);
    *direction /= *maxDistance;
  }
}

#endif
