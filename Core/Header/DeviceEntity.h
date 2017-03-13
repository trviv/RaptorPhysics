#ifndef RX_DEVICE_ENTITY
#define RX_DEVICE_ENTITY

#include "CudaQuery.h"

//base class for all the objects that will have a device
//counter part
//this class objects will be present in both host and device memory
CDEF template<class T>class DeviceEntity
{
  //protected:
  //Byte type;  //type identifier for the device entity

public:
  //default constructor to assign the type
  FORCE_INLINE CU_DEV_HOST DeviceEntity()
  {
    //this->type = type;
  }

  //function to export the device entity to device
  FORCE_INLINE CU_HOST void exportToDevice(T* device_ptr)
  {
    exportToDevice(this, device_ptr);
    CU_PROMPT
  }

  //function to export the device entity to device
  CU_HOST static void exportToDevice(const T* host_ptr, T* device_ptr,
    const int count = 1)
  {
    cudaMemcpy(device_ptr, host_ptr, sizeof(T)*count, cudaMemcpyHostToDevice);
    CU_PROMPT
  }

  //function to import from device to device entity
  CU_HOST static void importToHost(T* host_ptr, const T* device_ptr,
    const int count = 1)
  {
    cudaMemcpy(host_ptr, device_ptr, sizeof(T)*count, cudaMemcpyDeviceToHost);
    CU_PROMPT
  }

  //function to duplicate device data
  CU_HOST static void copy(T* dest_device_ptr, const T* src_device_ptr,
    const int count = 1)
  {
    cudaMemcpy(dest_device_ptr, src_device_ptr, sizeof(T)*count,
      cudaMemcpyDeviceToDevice);
    CU_PROMPT
  }

  //function to set device data
  CU_HOST static void set(T* dest_device_ptr, int value, const int count = 1)
  {
    cudaMemset(dest_device_ptr, value, sizeof(T)*count);
    CU_PROMPT
  }
};

CDEF template<class T>class DeviceEntityType : public DeviceEntity<T>
{
protected:
  Byte type;  //type identifier for the device entity

public:
  //default constructor to assign the type
  FORCE_INLINE CU_DEV_HOST DeviceEntityType(const Byte& type)
  {
    this->type = type;
  }
  /*
  //function to export the device entity to device
  FORCE_INLINE CU_HOST void exportToDevice(T* device_ptr)
  {
  cudaMemcpy(device_ptr, this, sizeof(T), cudaMemcpyHostToDevice);
  CU_PROMPT
  }
  */
};

//class to allocate a big array of a specific class
//this class instance will be present on host
CDEF template<class T>class DeviceAllocator
{
  T* device_ptr;    //pointer to device memory
  Counter elements; //the number of elements allocated

public:

  FORCE_INLINE DeviceAllocator()
    :device_ptr(NULL), elements(0)
  {}

  //free device memory while destroying object
  FORCE_INLINE ~DeviceAllocator()
  {
    free();
  }

  //assignemnt operator
  //it will just copy the meta data and not make a copy of actual data
  FORCE_INLINE void operator=(DeviceAllocator<T>& reference)
  {
    device_ptr = reference.device_ptr;
    elements = reference.elements;
  }

  //allocate device memory and return device memory address
  FORCE_INLINE T* alloc(const Counter& elements = 1)
  {
    //if memory of same size already allocated, use the same one
    if (this->elements == elements && device_ptr)
    {
      return device_ptr;
    }

    //if some memory already allocated
    if (this->elements != elements && device_ptr) free();

    //allocate new memory
    cudaMalloc((void**)&device_ptr, sizeof(T)*elements);
    CU_PROMPT

      this->elements = elements;
    return device_ptr;
  }

  //free allocated device memory
  FORCE_INLINE void free()
  {
    if (device_ptr)  //if some memory allocated
    {
      cudaFree(device_ptr);//free device memory
      CU_PROMPT
        device_ptr = NULL;
      elements = 0;
    }
  }

  FORCE_INLINE CU_DEV_HOST const Counter& size()const
  {
    return elements;
  }

  FORCE_INLINE CU_DEV_HOST T* get()
  {
    return device_ptr;
  }

  FORCE_INLINE const CU_DEV_HOST T* get()const
  {
    return device_ptr;
  }

  FORCE_INLINE const CU_DEV_HOST T& operator[](const Counter& index)const
  {
    return *(device_ptr + index);
  }

  FORCE_INLINE CU_DEV_HOST T& operator[](const Counter& index)
  {
    return *(device_ptr + index);
  }
};

//Macro to declare a device allocator member inside a class
//it will automatically provide member functions to access the declared member
#define DECLARE_DEVICE_ALLOCATOR(class_name, function_prefix) \
private: \
  DeviceAllocator<class_name> function_prefix##_alloc; \
public: \
  FORCE_INLINE CU_DEV_HOST const DeviceAllocator<class_name>& function_prefix##Allocator()const \
            { return function_prefix##_alloc;} \
  FORCE_INLINE CU_DEV_HOST DeviceAllocator<class_name>& function_prefix##Allocator() \
            { return function_prefix##_alloc;} \
private:

#endif