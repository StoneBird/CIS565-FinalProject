#version 330
out vec4 vFragColor;


void main() { 
 
 if(dot(gl_PointCoord-0.5,gl_PointCoord-0.5)>0.25)
 {
   discard;
 }
 else
 {
   vFragColor = vec4(1.0,1.0,0.0,1.0);  
 }
}