#include "Camera.h"

CameraDepth::CameraDepth(ComputeInterface* compute)
  :Camera(compute)
{
  focus = 0.f;
  aperture = 0.f;
}
