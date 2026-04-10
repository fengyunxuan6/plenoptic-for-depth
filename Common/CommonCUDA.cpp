/********************************************************************
file base:      CommonCUDA.cpp
author:         LZD
created:        2025/06/12
purpose:
*********************************************************************/
#include "Common/CommonCUDA.h"

#include <cstdarg>

namespace LFMVS
{
  void StringAppendV(std::string* dst, const char* format, va_list ap)
  {
    // First try with a small fixed size buffer.
    static const int kFixedBufferSize = 1024;
    char fixed_buffer[kFixedBufferSize];

    // It is possible for methods that use a va_list to invalidate
    // the data in it upon use.  The fix is to make a copy
    // of the structure before using it and use that copy instead.
    va_list backup_ap;
    va_copy(backup_ap, ap);
    int result = vsnprintf(fixed_buffer, kFixedBufferSize, format, backup_ap);
    va_end(backup_ap);

    if (result < kFixedBufferSize)
    {
        if (result >= 0)
        {
          // Normal case - everything fits.
          dst->append(fixed_buffer, result);
          return;
        }

  #ifdef _MSC_VER
      // Error or MSVC running out of space.  MSVC 8.0 and higher
      // can be asked about space needed with the special idiom below:
      va_copy(backup_ap, ap);
      result = vsnprintf(nullptr, 0, format, backup_ap);
      va_end(backup_ap);
  #endif

      if (result < 0)
      {
        // Just an error.
        return;
      }
    }

    // Increase the buffer size to the size requested by vsnprintf,
    // plus one for the closing \0.
    const int variable_buffer_size = result + 1;
    std::unique_ptr<char> variable_buffer(new char[variable_buffer_size]);

    // Restore the va_list before we use it again.
    va_copy(backup_ap, ap);
    result = vsnprintf(variable_buffer.get(), variable_buffer_size, format, backup_ap);
    va_end(backup_ap);

    if (result >= 0 && result < variable_buffer_size)
    {
      dst->append(variable_buffer.get(), result);
    }
  }

  std::string StringPrintf(const char* format, ...)
  {
    va_list ap;
    va_start(ap, format);
    std::string result;
    StringAppendV(&result, format, ap);
    va_end(ap);
    return result;
  }

  void CudaSafeCall(const cudaError_t error, const std::string& file,
                    const int line)
  {
    if (error != cudaSuccess) {
      std::cerr << StringPrintf("%s in %s at line %i", cudaGetErrorString(error),
                                file.c_str(), line)
                << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  void CudaCheckError(const char* file, const int line)
  {
      cudaError error = cudaGetLastError();
      if (error != cudaSuccess)
      {
          std::cerr << StringPrintf("cudaCheckError() failed at %s:%i : %s", file,
                                line, cudaGetErrorString(error)) << std::endl;
          exit(EXIT_FAILURE);
      }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    error = cudaDeviceSynchronize();
    if (cudaSuccess != error)
    {
      std::cerr << StringPrintf("cudaCheckError() with sync failed at %s:%i : %s",
                                file, line, cudaGetErrorString(error))
                << std::endl;
      std::cerr
          << "This error is likely caused by the graphics card timeout "
             "detection mechanism of your operating system. Please refer to "
             "the FAQ in the documentation on how to solve this problem."
          << std::endl;
      exit(EXIT_FAILURE);
    }
  }
}
