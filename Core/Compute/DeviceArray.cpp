#include "DeviceArray.h"

template<class ClassType> void DeviceArray<ClassType>::syncDevicePointerBuffer()
{
  if (!hostBuffer && !hostBuffer->size())
  {
    logComputeError("Device array does not have a host buffer!");
  }

  @autoreleasepool {
    MTLArgumentDescriptor* argumentDescriptor = [MTLArgumentDescriptor argumentDescriptor];

    argumentDescriptor.dataType = MTLDataTypePointer;
    argumentDescriptor.index    = 0;
    argumentDescriptor.access   = MTLArgumentAccessReadOnly;

    id <MTLArgumentEncoder> argumentEncoder = [compute->getDevice() newArgumentEncoderWithArguments:@[argumentDescriptor, ]];

    resize((uint) (host()->size() * argumentEncoder.encodedLength) / sizeof(ClassType), true);

    [argumentEncoder setArgumentBuffer:*deviceBuffer offset:0];

    for (uint i=0; i<host()->size(); i++)
    {
      if (typeid(ClassType) == typeid(const ComputeMemory*))
      {
        const ComputeMemory* memory = (const ComputeMemory*)host()->at(i);
        [argumentEncoder setBuffer:*memory offset:memory->getOffset() atIndex:i];
      }
    }
  }
}

template class DeviceArray<const ComputeMemory*>;
