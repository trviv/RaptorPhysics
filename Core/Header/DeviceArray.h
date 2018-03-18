#ifndef DEVICE_ARRAY_H
#define DEVICE_ARRAY_H

#include "ComputeInterface.h"

/*!
@class Class to allocate a device array of a specific type.
*/
template<class ClassType> class DeviceArray
{
  ComputeHeap*        heap;
  ComputeMemory*      deviceBuffer; //pointer to device memory
  ComputeInterface*   compute;
  vector<ClassType>*  hostBuffer;   //host memory
  uint  elements;                   //the number of elements allocated
  uint  allocated;
  bool  shared;                     //if the array is shared

  void allocDevice()
  {
    //allocate new memory
    deviceBuffer = heap->alloc(sizeof(ClassType)*elements);
    allocated = elements;
  }

  void freeDevice()
  {
    if (deviceBuffer)
    {
      heap->free(deviceBuffer);
      deviceBuffer = NULL;
    }
  }

public:

  DeviceArray()
  {
    heap = NULL;
    deviceBuffer = NULL;
    compute = NULL;
    hostBuffer = NULL;
    elements = 0;
    allocated = 0;
    shared = false;
  }

  DeviceArray(ComputeInterface* compute, ComputeHeap* heap = NULL, bool shared = false)
    : DeviceArray()
  {
    create(compute, heap, shared);
  }

  DeviceArray(const DeviceArray& ref)
  {
    logComputeError("Copying device array is not defined yet!");
  }

  void create(ComputeInterface* compute, ComputeHeap* heap = NULL, bool shared = false)
  {
    free();
    deviceBuffer = NULL;
    hostBuffer = NULL;
    elements = 0;
    allocated = 0;
    this->compute = compute;
    this->heap = heap ? heap : (compute ? &compute->heap : NULL);
    this->shared = shared;

    if (shared)
    {
      hostBuffer = new vector<ClassType>();
    }
  }

  //free device memory while destroying object
  ~DeviceArray()
  {
    free();
  }

  /*//it will just copy the meta data and not make a copy of actual data
  void assign(DeviceArray<ClassType>& reference, uint elementOffset = 0)
  {
  free();
  compute = reference.compute;
  if (elementOffset)
  {
  deviceBuffer = compute->offsetMemory(reference.deviceBuffer, elementOffset*sizeof(ClassType));
  }
  else
  {
  deviceBuffer = reference.deviceBuffer;
  }
  hostBuffer = reference.hostBuffer;
  elements = reference.elements - elementOffset;
  offset = elementOffset;
  shared = reference.shared;
  ownsMemory = false;
  }*/

  //allocate device memory and return device memory address
  /*void alloc(const uint elements = 1)
  {
  //if memory of same size already allocated, use the same one
  if (this->elements == elements && deviceBuffer)
  {
  return;
  }

  //if some memory already allocated
  if (this->elements != elements && deviceBuffer) free();

  //allocate new memory
  compute->alloc(sizeof(ClassType)*elements);
  this->elements = elements;
  ownsMemory = true;
  }*/

  void resize(const uint elements, bool copyOld)
  {
    /*if (shared && elements != hostBuffer->size())
    {
    (*hostBuffer).resize(elements);
    }*/
    if (elements <= allocated)
    {
      this->elements = elements;
      return;
    }

    ComputeMemory* newArray = heap->alloc(sizeof(ClassType)*elements);
    if (deviceBuffer)
    {
      if (copyOld)
      {
        compute->copyBuffer(deviceBuffer, newArray, 0, 0, sizeof(ClassType)*this->elements);
      }
      freeDevice();
    }
    deviceBuffer = newArray;
    allocated = elements;
    this->elements = elements;
  }

  //free allocated device memory
  void free()
  {
    freeDevice();
    if (hostBuffer)
    {
      delete hostBuffer;
      hostBuffer = NULL;
    }
    elements = 0;
    allocated = 0;
  }

  void syncHost(size_t offset = 0, size_t elements = 0)
  {
    if (!hostBuffer)
    {
      logComputeError("Device array does not have a host buffer!");
    }
    if (!elements)
    {
      elements = this->elements;
    }
    if ((offset + elements) == 0)
    {
      logComputeError("Device array is empty!");
    }
    //if ((offset + size) != elements)
    {
      hostBuffer->resize(offset + elements);
    }
    compute->copyToHost(deviceBuffer, offset * sizeof(ClassType), elements * sizeof(ClassType), &((*hostBuffer)[offset]), false);
  }

  void syncDevice(size_t offset = 0, size_t size = 0)
  {
    if (!hostBuffer)
    {
      logComputeError("Device array does not have a host buffer!");
    }
    if (!size)
    {
      size = hostBuffer->size();
    }
    if ((offset + size) != elements)
    {
      resize((offset + size), true);
    }
    if (hostBuffer->size())
    {
      compute->copyFromHost(deviceBuffer, offset * sizeof(ClassType), size * sizeof(ClassType), &((*hostBuffer)[offset]), false);
    }
    else
    {
      logComputeMessage("Nothing to copy from host!");
    }
  }

  uint size()const
  {
    return elements;
  }

  vector<ClassType>* host()
  {
    return hostBuffer;
  }

  const vector<ClassType>* host()const
  {
    return hostBuffer;
  }

  uint hostOffset()const
  {
    return offset;
  }

  ComputeMemory* device()
  {
    return deviceBuffer;
  }

  const ComputeMemory* device()const
  {
    return deviceBuffer;
  }
};

#endif