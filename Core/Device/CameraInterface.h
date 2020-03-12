#ifndef CAMERA_INTERFACE_H
#define CAMERA_INTERFACE_H

#include "../Header/ComputeInterface.h"

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

  const ComputeMemory* getCurrentFrame()const;
};

#endif
