cmake_minimum_required(VERSION 3.10)
project(mergemask LANGUAGES C VERSION 0.0.1)
add_executable(mm merge.c)
target_compile_options(mm PRIVATE -mpclmul)
