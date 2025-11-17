/*
 * Copyright (c) Vivek Trivedi, 2025
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#ifndef CAMERA_INTERFACE_H
#define CAMERA_INTERFACE_H

#include <Compute/ComputeInterface.h>

/*!
@class Interface providing abstraction to access Camera.
*/
class CameraInterface
{
public:
  /*!@constructor Create a new camera interface, compute interface is required for possible device query.*/
  CameraInterface(ComputeInterface* compute);

  ~CameraInterface();

  void startSession();

  bool isActive()const;

  uint width()const;

  uint height()const;

  uint bytesPerPixel()const;

  const ComputeTexture* getCurrentFrame()const;
};

#endif
