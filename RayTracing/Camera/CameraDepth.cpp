/*
 * RaptorPhysics
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#include "Camera.h"

CameraDepth::CameraDepth(ComputeInterface* compute)
  :Camera(compute)
{
  focus = 0.f;
  aperture = 0.f;
}
