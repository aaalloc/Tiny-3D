project('imgui','cpp')


sources = [
    'imconfig.h',
    'imgui_demo.cpp',
    'imgui_draw.cpp',
    'imgui_impl_sdl2.cpp',
    'imgui_impl_sdl2.h',
    'imgui_impl_vulkan.cpp',
    'imgui_impl_vulkan.h',
    'imgui_internal.h',
    'imgui_tables.cpp',
    'imgui_widgets.cpp',
    'imgui.cpp',
    'imgui.h',
    'imstb_rectpack.h',
    'imstb_textedit.h',
    'imstb_truetype.h',
]

subproject('volk')

library(
    meson.project_name(),
    sources,
    dependencies: [
        dependency('vulkan'),
        dependency('sdl2'),
    ],
    install: true
)