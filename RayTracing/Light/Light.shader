#ifndef LIGHT_SHADER
#define LIGHT_SHADER

inline void sampleLight(Thread float3* position, Thread float3* color, Const LightStruct* light, const uint offset, const ushort bounce)
{
  *position = light->position;
  *color = light->color;
  if (getRayTracingEntityType(light->identity) == 2)
  {
    const float2 random = constructFloat2(getRandomNumber(offset, 2+bounce*4+0), getRandomNumber(offset, 2+bounce*4+1)) * 2.0f - 1.0f;
    //const half2 random = half2(getRandomNumber(offset, 2+bounce*4+0), getRandomNumber(offset, 2+bounce*4+1)) * 2.0f - 1.0f;
    *position += constructFloat3(light->right.xyz) * random.x + constructFloat3(light->up.xyz) * random.y;
  }
}

#endif
