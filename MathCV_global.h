#ifndef MATHCV_GLOBAL_H
#define MATHCV_GLOBAL_H

// Для Windows: dllexport/dllimport
#if defined(_WIN32)
    #if defined(MATHCV_LIBRARY)
        #define MATHCV_EXPORT __declspec(dllexport)
    #else
        #define MATHCV_EXPORT __declspec(dllimport)
    #endif
#else
    // Для Linux/macOS: visibility attributes
    #if defined(MATHCV_LIBRARY)
        #define MATHCV_EXPORT __attribute__((visibility("default")))
    #else
        #define MATHCV_EXPORT
    #endif
#endif

#endif // MATHCV_GLOBAL_H