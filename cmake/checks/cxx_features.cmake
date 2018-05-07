# first see if the compiler accepts the attribute
check_cxx_source_compiles(
  "
          int old_fn () __attribute__((deprecated));
          int old_fn () { return 0; }
          int (*fn_ptr)() = old_fn;

          int main () {}
  "
  MY_COMPILER_HAS_ATTRIBUTE_DEPRECATED
  )

IF(MY_COMPILER_HAS_ATTRIBUTE_DEPRECATED)
  SET(MY_DEPRECATED "__attribute__((deprecated))" CACHE INTERNAL "" FORCE)
ELSE()
  SET(MY_DEPRECATED " " CACHE INTERNAL "" FORCE)
ENDIF()

set(CMAKE_REQUIRED_FLAGS "-std=c++14")
check_cxx_source_compiles(
  "
           auto foo(int a)
           {
             return 1;
           }

           int main () {}
  "
  HAVE_CXX14_RETURN_AUTO
  )
