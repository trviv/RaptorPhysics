/*
 * Copyright (c) Vivek Trivedi, 2025
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#ifndef CLOCK_H
#define CLOCK_H

class Clock
{
public:
  ///The b3Clock is a portable basic clock that measures accurate time in seconds, use for profiling.
  Clock();

  Clock(const Clock& other);

  Clock& operator=(const Clock& other);

  ~Clock();

  /// Resets the initial reference time.
  void reset();

  /// Returns the time in ms since the last call to reset or since 
  /// the b3Clock was created.
  /// Returns the time in ms since the last call to reset or since 
  /// the b3Clock was created.
  unsigned long int getTimeMilliseconds();

  /// Returns the time in us since the last call to reset or since 
  /// the Clock was created.
  unsigned long int getTimeMicroseconds();

private:
  struct ClockData* m_data;
};

#endif