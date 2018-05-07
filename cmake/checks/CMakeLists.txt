FILE(GLOB_RECURSE files "cmake/checks/*cmake")
foreach(var ${files})
  include("${var}")
endforeach(var)
