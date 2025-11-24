/*
 * RaptorPhysics
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#ifndef MEMORY_MANAGER_H
#define MEMORY_MANAGER_H

#include <Compute/ComputeShared.h>
#include <Compute/ComputeInterface.h>
#include <unordered_map>

/*!
@class Base class to all objects wanting to use memory manager.
*/
class MemoryManaged
{
  friend class MemoryManager;

public:

  bool persistant = false;

  /*!@function This function should implement memory cleaning for objects allocated by memory manager.*/
  virtual void freeManaged()const = 0;
};

/*!
@class Class to allocate and automatically deallocate memory.
*/
class MemoryManager
{
  unordered_map<const MemoryManaged*, int> allocatedMemory;
public:

  MemoryManager();

  ~MemoryManager();

  /*!@function Allocate memory of the specified size using keeping entity pointer as the reference.*/
  void* alloc(const MemoryManaged* managed, uint sizeInBytes);

  /*!@function Deallocate unused memory, this function should be called only after a certain number of frames.*/
  void dealloc();

  /*!@function Register a hit to the memory managed object.*/
  void hit(const MemoryManaged* managed);
};

extern MemoryManager memoryManager;

#endif
