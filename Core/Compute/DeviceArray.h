#ifndef DEVICE_ARRAY_H
#define DEVICE_ARRAY_H

#include <Utils/MemoryManager.h>

/*!
@class Class to allocate a device array of a specific type.
*/
template<class ClassType> class DeviceArray : MemoryManaged
{
  ComputeHeap*        heap;
  ComputeMemory*      deviceBuffer; //pointer to device memory
  ComputeInterface*   compute;
  mutable vector<ClassType>*  hostBuffer; //host memory
  uint  elements; //the number of elements allocated
  uint  allocated;

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

  void freeManaged()const
  {
    if (hostBuffer)
    {
      hostBuffer->clear();
      delete hostBuffer;
      hostBuffer = NULL;
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
    persistant = false;
  }

  DeviceArray(ComputeInterface* compute, ComputeHeap* heap = NULL, bool persistantHeap = false) : DeviceArray()
  {
    create(compute, heap, persistantHeap);
  }

  DeviceArray(const DeviceArray& ref)
  {
    logComputeError("Copying device array is not defined yet!");
  }

  void create(ComputeInterface* compute, ComputeHeap* heap = NULL, bool persistantHeap = false)
  {
    if (this->compute)
    {
      logComputeError("Trying to re create device array!");
    }

    free();
    deviceBuffer = NULL;
    hostBuffer = NULL;
    elements = 0;
    allocated = 0;
    persistant = persistantHeap;
    this->compute = compute;
    this->heap = heap ? heap : (compute ? &compute->heap : NULL);
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
    freeManaged();
    elements = 0;
    allocated = 0;
  }

  void syncHost(size_t hostOffset = 0, size_t elements = 0, size_t deviceOffset = 0)const
  {
    if (!elements)
    {
      elements = this->elements;
    }
    if ((deviceOffset + elements) == 0)
    {
      logComputeError("Device array is empty!");
    }

    if (host()->size() < (hostOffset + elements))
    {
      hostBuffer->resize(hostOffset + elements);
    }
    compute->copyToHost(deviceBuffer, deviceOffset * sizeof(ClassType), elements * sizeof(ClassType), &((*hostBuffer)[hostOffset]), false);
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
      resize((uint)(offset + size), true);
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

  void syncDevice(const PartitionInfo& partition)
  {
    syncDevice(partition.offset, partition.count);
  }

  uint size()const
  {
    return elements;
  }

  vector<ClassType>* host()
  {
    if (!hostBuffer)
    {
      hostBuffer = (vector<ClassType>*)memoryManager.alloc(this, sizeof(vector<ClassType>));
      new (hostBuffer) vector<ClassType>();
    }
    else
    {
      memoryManager.hit(this);
    }
    return hostBuffer;
  }

  const vector<ClassType>* host()const
  {
    if (!hostBuffer)
    {
      hostBuffer = (vector<ClassType>*)memoryManager.alloc(this, sizeof(vector<ClassType>));
      new (hostBuffer) vector<ClassType>();
    }
    else
    {
      memoryManager.hit(this);
    }
    return hostBuffer;
  }

  ComputeMemory* device()
  {
    return deviceBuffer;
  }

  const ComputeMemory* device()const
  {
    return deviceBuffer;
  }

  void syncDevicePointerBuffer(const uint deviceByteOffset = 0);
};

#endif
