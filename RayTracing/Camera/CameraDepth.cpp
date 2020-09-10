#include "Camera.h"

CameraDepth::CameraDepth(ComputeInterface* compute)
  :CameraSimple(compute)
{
  focus = 0.f;
  aperture = 0.f;
}
