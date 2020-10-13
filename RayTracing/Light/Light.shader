#ifndef LIGHT_SHADER
#define LIGHT_SHADER

void sampleLight(Thread float3* position, Thread float3* color, const LightStruct light)
{
  *position = light.position;
  *color    = light.color;
}

#endif
