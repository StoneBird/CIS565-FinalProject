#version 330

in vec3 position;



uniform mat4 u_projMatrix;
uniform mat4 u_modelView;

void main(){
   gl_PointSize = 10;                                                                                                                     

   //gl_Position = u_projMatrix * u_modelView * vec4(position, 1.0);
   gl_Position = vec4(position,1.0);
}