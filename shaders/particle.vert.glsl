#version 330

layout(location = 0) in vec3 position;



uniform mat4 u_projMatrix;
uniform mat4 u_modelView;
uniform vec2 u_screenSize;
uniform float u_spriteSize;

void main(){
    vec4 eyePos = u_modelView * vec4(position,1.0);
    vec4 projVoxel = u_projMatrix * vec4(u_spriteSize,u_spriteSize,eyePos.z,eyePos.w);
    vec2 projSize = u_screenSize * projVoxel.xy / projVoxel.w;
    gl_PointSize = 0.25 * (projSize.x+projSize.y);
    gl_Position = u_projMatrix * eyePos;
    
    
}