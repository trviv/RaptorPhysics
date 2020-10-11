#ifndef LIGHT_SHADER
#define LIGHT_SHADER

void sampleLight(
  constantKernelInput(LightStruct, light)
  KERNEL_GLOBAL_ARGUMENTS)
{
}

#endif
