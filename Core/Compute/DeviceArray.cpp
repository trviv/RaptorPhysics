/*
 * RaptorPhysics
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#include "DeviceArray.h"

#ifdef USE_METAL_COMPUTE

template<class ClassType> void DeviceArray<ClassType>::syncDevicePointerBuffer(const uint deviceByteOffset)
{
  if (!hostBuffer || !hostBuffer->size())
  {
    logComputeError("Device array does not have a host buffer!");
  }

  if (typeid(ClassType) != typeid(const ComputeMemory*))
  {
    logComputeError("syncDevicePointerBuffer is only available with type ComputeMemory*!");
  }

  @autoreleasepool {
    MTLArgumentDescriptor* argumentDescriptor = [MTLArgumentDescriptor argumentDescriptor];

    argumentDescriptor.index = 0;
    argumentDescriptor.access = MTLBindingAccessReadOnly;
    argumentDescriptor.dataType = MTLDataTypePointer;
    argumentDescriptor.arrayLength = host()->size();

    id <MTLArgumentEncoder> argumentEncoder = [compute->getDevice() newArgumentEncoderWithArguments:@[argumentDescriptor, ]];

    resize((uint)((host()->size() + deviceByteOffset) * argumentEncoder.encodedLength) / sizeof(ClassType), false);

    [argumentEncoder setArgumentBuffer:*(device()) offset:device()->getOffset() + deviceByteOffset];

    for (uint i=0; i<host()->size(); i++)
    {
      const ComputeMemory* memory = (const ComputeMemory*)host()->at(i);
      [argumentEncoder setBuffer:*memory offset:memory->getOffset() atIndex:i];
    }
  }
}

#else

template<class ClassType> void DeviceArray<ClassType>::syncDevicePointerBuffer(const uint deviceByteOffset)
{
  logComputeError("syncDevicePointerBuffer function is not implemented!");
}

#endif

template class DeviceArray<const ComputeMemory*>;
