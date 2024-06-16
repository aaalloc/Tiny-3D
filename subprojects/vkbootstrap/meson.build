project('vkbootstrap','cpp')


sources = [
    'VkBootstrap.cpp',
    'VkBootstrap.h',
    'VkBootstrapDispatch.h',
]

vkboostrap_lib = library(
    meson.project_name(),
    sources,
    install: true
)

vkbootstrap_dep= declare_dependency(
    link_with: vkboostrap_lib,
    include_directories: include_directories('.')
)