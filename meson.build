project('vulkan_guide', 
    'cpp', 
    default_options: ['default_library=static', 'warning_level=2'],
)

fs = import('fs') 

subproject('vkbootstrap')

project_dependencies = [
    dependency('vulkan', required: true),
    dependency('fastgltf', required: true, fallback: ['fastgltf', 'fastgltf_dep']),
    dependency('glm', required: true, fallback: ['glm', 'glm_dep']),
    dependency('fmt', required: true, fallback: ['fmt', 'fmt_dep']),
    dependency('sdl2', required: true, fallback: ['sdl2', 'sdl2_dep']),
    dependency('stb_image'),
    dependency('vkbootstrap'),
    dependency('vma'),
    dependency('imgui'),
]

shader_src = meson.current_source_dir() / 'shaders'
shader_globs = [
    'shaders/gradient.comp',
    'shaders/sky.comp',
    'shaders/gradient_color.comp',
]

glsl_validator = find_program('glslangValidator', dirs: [
    '/usr/bin',
    '/usr/local/bin',
])

shaders = []

foreach pattern : shader_globs
    shaders += files(pattern)
endforeach

spirv_outputs = []


foreach shader : shaders
    shader_name = fs.name(shader)
    spirv = '@0@.spv'.format(shader_name)
    spirv_outputs += custom_target(
        shader_name,
        output: spirv,
        command: [glsl_validator, '-V', '@INPUT@', '-o', '@OUTPUT@'],
        input: shader,
        build_by_default: true
    )
endforeach

custom_target(
    'shaders',
    input: spirv_outputs,
    output: 'shaders',
    command: ['echo', 'Generated shaders']
)


executable(
    'engine',
    [
        'src/main.cpp',
        'src/vk_initializers.cpp',
        'src/vk_images.cpp',
        'src/vk_descriptors.cpp',
        'src/vk_pipelines.cpp',
        'src/vk_engine.cpp',
        'src/vk_loader.cpp',
        'src/camera.cpp',
    ],
    override_options: ['cpp_std=c++20'],
    include_directories: [
        include_directories('include')
    ],
    dependencies: project_dependencies,
    cpp_pch: 'pch/vk_render.h',
    cpp_args: ['-Wno-missing-field-initializers', '-Wno-missing-braces', '-Wno-unused-parameter']
)