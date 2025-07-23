
#ifndef LOGGING_EXPORT_H
#define LOGGING_EXPORT_H

#ifdef LOGGING_STATIC_DEFINE
#  define LOGGING_EXPORT
#  define LOGGING_NO_EXPORT
#else
#  ifndef LOGGING_EXPORT
#    ifdef logging_EXPORTS
        /* We are building this library */
#      define LOGGING_EXPORT 
#    else
        /* We are using this library */
#      define LOGGING_EXPORT 
#    endif
#  endif

#  ifndef LOGGING_NO_EXPORT
#    define LOGGING_NO_EXPORT 
#  endif
#endif

#ifndef LOGGING_DEPRECATED
#  define LOGGING_DEPRECATED __declspec(deprecated)
#endif

#ifndef LOGGING_DEPRECATED_EXPORT
#  define LOGGING_DEPRECATED_EXPORT LOGGING_EXPORT LOGGING_DEPRECATED
#endif

#ifndef LOGGING_DEPRECATED_NO_EXPORT
#  define LOGGING_DEPRECATED_NO_EXPORT LOGGING_NO_EXPORT LOGGING_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef LOGGING_NO_DEPRECATED
#    define LOGGING_NO_DEPRECATED
#  endif
#endif

#endif /* LOGGING_EXPORT_H */
