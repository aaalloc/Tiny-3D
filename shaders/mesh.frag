#version 450

#extension GL_GOOGLE_include_directive : require
#include "input_structures.glsl"

layout(location = 0) in vec3 inNormal;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec3 inFragPos; // Fragment position in world space

layout(location = 0) out vec4 outFragColor;

float specularStrength = 2.f;
vec3 lightColor = vec3(1.0f);
vec3 calcPointLight(Light light, vec3 inFragPos, vec3 inNormal)
{
    vec3 lightDir = normalize(light.position.xyz - inFragPos);
    float diff = max(dot(lightDir, inNormal), 0.0f);

    vec3 viewDir = normalize(sceneData.cameraPosition.xyz - inFragPos);
    vec3 reflectDir = reflect(viewDir, inNormal);

    float spec = pow(max(dot(viewDir, reflectDir), 0.0f), 32.0f);
    vec3 specular = specularStrength * spec * lightColor;
    vec3 ambient = sceneData.ambientColor.xyz;

    // specular needs some work, so it is not used for now
    return (ambient + diff) * light.power;
}

void main()
{
    vec3 result = calcPointLight(sceneData.lights[0], inFragPos, inNormal);
    vec3 color = inColor * texture(colorTex, inUV).xyz;
    outFragColor = vec4(result * color, 1.0f);
}