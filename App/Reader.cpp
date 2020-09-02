#include "Reader.h"

void Reader::readIntArray(int* dest, const char* source)
{
  for (char number = *source; number;)
  {
    // skim through all the spaces present in between
    if (number == ' ' || number == '\n' || number == '\t' || number == '\0')
    { source++; number = *source; continue;}
    *dest = atoi(source);
    dest++;
    // skim through all the digits
    while ((number >= '0' && number <= '9') || number == '-' || number == '+')
    { source++; number = *source;}
  }
}

void Reader::readFloatArray(float* dest, const char* source)
{
  for (char number = *source; number;)
  {
    // skim through all the spaces present in between
    if (number == ' ' || number == '\n' || number == '\t' || number == '\0')
    { source++; number = *source; continue;}
    *dest = atof(source);
    dest++;
    // skim through all the digits
    while ((number >= '0' && number <= '9') || number == '.' || number == '-' || number == '+' || number == 'e')
    { source++; number = *source;}
  }
}
