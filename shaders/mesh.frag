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
    vec3 lightDir = normalize(light.position - inFragPos);
    float diff = max(dot(lightDir, inNormal), 0.0f);

    vec3 viewDir = normalize(sceneData.cameraPosition.xyz - inFragPos);
    vec3 reflectDir = reflect(viewDir, inNormal);

    float spec = pow(max(dot(viewDir, reflectDir), 0.0f), 32.0f);
    vec3 specular = specularStrength * spec * lightColor;
    vec3 ambient = sceneData.ambientColor.xyz;

    // specular needs some work, so it is not used for now
    return (ambient + diff) * light.power;
}

// Light lights[2] = Light[2](Light(vec3(62.0, -35.0, -28.0), 1.0),
//                            Light(vec3(13.685, -23.596, -71.821), 1.0));
void main()
{
    vec3 result = vec3(0.0f);
    for (int i = 0; i < 2; i++)
        result += calcPointLight(sceneData.lights[i], inFragPos, inNormal);
    vec3 color = inColor * texture(colorTex, inUV).xyz;
    outFragColor = vec4(result * color, 1.0f);
}