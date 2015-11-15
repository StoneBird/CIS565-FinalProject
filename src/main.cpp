/**
 * @file      main.cpp
 * @brief     Main file for CUDA rendering. Handles CUDA-GL interop for display.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya
 * @date      2012-2015
 * @copyright University of Pennsylvania
 */

#include "main.hpp"
#include <chrono>

#include "RigidBody.h"
#include "particleSampling.h"

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "Usage: [obj file]" << endl;
        return 0;
    }

	//GL init
	frame = 0;
	seconds = time(NULL);
	fpstracker = 0;

	if (samplingTest_Init()) {
		
		//test: rigid body sampling
		//RigidBody rigid_body;

		//rigid_body.initObj(argv[1]);

		//rigid_body.initParticles(10);


		// GLFW main loop
		//mainLoop();
		
		samplingTest_Loop();
	}

	

	
	return 0;
}




void samplingTest_Loop()
{

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		
		time_t seconds2 = time(NULL);

		frame++;
		fpstracker++;

		if (seconds2 - seconds >= 1) {

			fps = fpstracker / (seconds2 - seconds);
			fpstracker = 0;
			seconds = seconds2;
		}

		string title = "CIS565 Final | " + utilityCore::convertIntToString((int)fps) + " FPS";
		glfwSetWindowTitle(window, title.c_str());


		//update camera
		updateCamera();
		

		
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		//glClear(GL_COLOR_BUFFER_BIT);

		//std::cout << glewGetErrorString(glGetError()) << '\n';
		
		
		glUseProgram(program);
		//float tmp[16] = { 1.0, 0.0, 0.0, 0.0,
		//	0.0, 1.0, 0.0, 0.0,
		//	0.0, 0.0, 1.0, 0.0,
		//	0.0, 0.0, 0.0, 1.0 };

		glm::vec2 screenSize(width, height);
		glm::mat4 modelView = view * model;
		glUniformMatrix4fv(u_modelView, 1, GL_FALSE, &modelView[0][0]);
		glUniformMatrix4fv(u_projMatrix, 1, GL_FALSE, &projection[0][0]);
		glUniform2fv(u_screenSize, 1, &screenSize[0]);
		glUniform1f(u_spriteSize, 0.1*scale);

		glUniform3f(u_color, 1.0, 1.0, 0.0);
		glUniform3f(u_lightDir, 0.0, 0.0, 1.0);

		glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

		//glPointSize(10);
		glDrawArrays(GL_POINTS, 0, num_points);
		glDisableVertexAttribArray(0);
		
		glUseProgram(0);

		glfwSwapBuffers(window);

	}
	glfwDestroyWindow(window);
	glfwTerminate();
}



//sampling Test init

bool samplingTest_Init()
{
	//GL window
	glfwSetErrorCallback(errorCallback);

	if (!glfwInit()) {
		return false;
	}

	width = 800;
	height = 800;

	
	
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	
	window = glfwCreateWindow(width, height, "Particles Simulation", NULL, NULL);
	if (!window) {
		fprintf(stderr, "Failed to open GLFW window.\n");
		glfwTerminate();
		return false;
	}
	glfwMakeContextCurrent(window);		//Initialize GLEW
	glfwSetKeyCallback(window, keyCallback);
	//MY Mouse Control
	glfwSetMouseButtonCallback(window, mouseButtonCallback);
	glfwSetCursorPosCallback(window, mouseMotionCallback);
	glfwSetScrollCallback(window, mouseWheelCallback);

	// Set up GL context
	glewExperimental = GL_TRUE;			//Needed in core profile
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		return false;
	}

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	samplingTest_InitVAO();

	samplingTest_InitShaders(program);

	return true;
}

void samplingTest_InitVAO()
{
	//GLfloat vertices[] = {
	//	-1.0f, -1.0f,
	//	1.0f, -1.0f,
	//	1.0f, 1.0f,
	//	-1.0f, 1.0f,
	//};

	//GLfloat texcoords[] = {
	//	1.0f, 1.0f,
	//	0.0f, 1.0f,
	//	0.0f, 0.0f,
	//	1.0f, 0.0f
	//};

	//GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

	////GLuint vertexBufferObjID[3];
	//glGenBuffers(3, vertexBufferObjID);

	//glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
	//glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	//glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
	//glEnableVertexAttribArray(positionLocation);

	//glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
	//glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
	//glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
	//glEnableVertexAttribArray(texcoordsLocation);

	//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
	//glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);



	//GLfloat g_vertex_buffer_data[] = {
	//	-0.5f, -0.5f, 0.0f,
	//	0.5f, -0.5f, -1.0f,
	//	0.5f, 0.5f, 0.0f,
	//	-0.5f, 0.5f, 1.0f
	//};

	GLfloat g_vertex_buffer_data[] = {
		-0.5f, 0.0f, -0.2f,
		-0.2f, 0.0f, -0.2f,
		-0.7f, 0.0f, -0.2f,
		0.1f, 0.0f, -0.2f,
		0.3f, 0.0f, -0.2f,
		0.6f, 0.0f, -0.2f,

		-0.5f, 0.0f, 0.0f,
		-0.2f, 0.0f, 0.0f,
		-0.7f, 0.0f, 0.0f,
		0.1f, 0.0f, 0.0f,
		0.3f, 0.0f, 0.0f,
		0.6f, 0.0f, 0.0f,

		-0.5f, 0.0f, 0.5f,
		-0.2f, 0.0f, 0.5f,
		-0.7f, 0.0f, 0.5f,
		0.1f, 0.0f, 0.5f,
		0.3f, 0.0f, 0.5f,
		0.6f, 0.0f, 0.5f,

		0.5f, -0.5f, -1.0f,
		0.5f, 0.5f, 0.0f,
		-0.5f, 0.5f, 1.0f
	};

	GLuint VertexArrayID[1];
	glGenVertexArrays(1, VertexArrayID);
	glBindVertexArray(VertexArrayID[0]);


	glGenBuffers(1, &vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	num_points = sizeof(g_vertex_buffer_data)/3/4;
}


void samplingTest_InitShaders(GLuint & program) {
	GLint location;

	program = glslUtility::createProgram(
		"../shaders/particle.vert.glsl",
		"../shaders/particle.frag.glsl",
		samplingTest_attributeLocations, 1);
	glUseProgram(program);

	//if ((location = glGetUniformLocation(program, "u_projMatrix")) != -1) {
	//	glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
	//}
	//if ((location = glGetUniformLocation(program, "u_cameraPos")) != -1) {
	//	glUniform3fv(location, 1, &cameraPosition[0]);
	//}

	glEnable(GL_PROGRAM_POINT_SIZE);


	//vertex shader
	u_modelView = glGetUniformLocation(program, "u_modelView");
	u_projMatrix = glGetUniformLocation(program, "u_projMatrix");
	u_screenSize = glGetUniformLocation(program, "u_screenSize");
	u_spriteSize = glGetUniformLocation(program, "u_spriteSize");


	//fragment shader
	u_color = glGetUniformLocation(program, "u_color");
	u_lightDir = glGetUniformLocation(program, "u_lightDir");
}

///////////////////////////////////////////////////////















//----------------------------------------------
//---------OLD CUDA rasterizer pipeline---------
//----------------------------------------------


//old rasterizer main loop
//render a 2d rect, texture is given by cuda
void mainLoop() {
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        runCuda();

        time_t seconds2 = time (NULL);

        if (seconds2 - seconds >= 1) {

            fps = fpstracker / (seconds2 - seconds);
            fpstracker = 0;
            seconds = seconds2;
        }

        string title = "CIS565 Final | " + utilityCore::convertIntToString((int)fps) + " FPS";
        glfwSetWindowTitle(window, title.c_str());

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, displayImage);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glClear(GL_COLOR_BUFFER_BIT);

        // VAO, shader program, and texture already bound
        glDrawElements(GL_TRIANGLES, 6,  GL_UNSIGNED_SHORT, 0);
        glfwSwapBuffers(window);
    }
    glfwDestroyWindow(window);
    glfwTerminate();
}

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

void runCuda() {
    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer
    dptr = NULL;

    cudaGLMapBufferObject((void **)&dptr, pbo);
	//rasterize(dptr);
    cudaGLUnmapBufferObject(pbo);

    frame++;
    fpstracker++;

}

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

bool init() {

	//GL window
    glfwSetErrorCallback(errorCallback);

    if (!glfwInit()) {
        return false;
    }

    width = 800;
    height = 800;
    window = glfwCreateWindow(width, height, "Particles Simulation", NULL, NULL);
    if (!window) {
		fprintf(stderr, "Failed to open GLFW window.\n");
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);		//Initialize GLEW
	glfwSetKeyCallback(window, keyCallback);

    // Set up GL context
    glewExperimental = GL_TRUE;			//Needed in core profile
    if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
        return false;
    }


	

    initVAO();
    initTextures();
    initCuda();
    initPBO();

    

    GLuint passthroughProgram;
    passthroughProgram = initShader();

    glUseProgram(passthroughProgram);
    glActiveTexture(GL_TEXTURE0);

    return true;
}

void initPBO() {
    // set up vertex data parameter
    int num_texels = width * height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;

    // Generate a buffer ID called a PBO (Pixel Buffer Object)
    glGenBuffers(1, &pbo);

    // Make this the current UNPACK buffer (OpenGL is state-based)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

    // Allocate data for the buffer. 4-channel 8-bit image
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
    cudaGLRegisterBufferObject(pbo);

}

void initCuda() {
    // Use device with highest Gflops/s
    cudaGLSetGLDevice(0);

	//rasterizeInit(width, height);

    // Clean up on program exit
    atexit(cleanupCuda);
}

void initTextures() {
    glGenTextures(1, &displayImage);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA,
                  GL_UNSIGNED_BYTE, NULL);
}

void initVAO(void) {
    GLfloat vertices[] = {
        -1.0f, -1.0f,
        1.0f, -1.0f,
        1.0f,  1.0f,
        -1.0f,  1.0f,
    };

    GLfloat texcoords[] = {
        1.0f, 1.0f,
        0.0f, 1.0f,
        0.0f, 0.0f,
        1.0f, 0.0f
    };

    GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

    GLuint vertexBufferObjID[3];
    glGenBuffers(3, vertexBufferObjID);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(positionLocation);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(texcoordsLocation);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}


GLuint initShader() {
    const char *attribLocations[] = { "Position", "Tex" };
    GLuint program = glslUtility::createDefaultProgram(attribLocations, 2);
    GLint location;

    //glUseProgram(program);
    if ((location = glGetUniformLocation(program, "u_image")) != -1) {
        glUniform1i(location, 0);
    }

    return program;
}

//-------------------------------
//---------CLEANUP STUFF---------
//-------------------------------

void cleanupCuda() {
    if (pbo) {
        deletePBO(&pbo);
    }
    if (displayImage) {
        deleteTexture(&displayImage);
    }
}

void deletePBO(GLuint *pbo) {
    if (pbo) {
        // unregister this buffer object with CUDA
        cudaGLUnregisterBufferObject(*pbo);

        glBindBuffer(GL_ARRAY_BUFFER, *pbo);
        glDeleteBuffers(1, pbo);

        *pbo = (GLuint)NULL;
    }
}

void deleteTexture(GLuint *tex) {
    glDeleteTextures(1, tex);
    *tex = (GLuint)NULL;
}

void shut_down(int return_code) {
	//rasterizeFree();
	samplingFree();	//TODO:test

    cudaDeviceReset();
#ifdef __APPLE__
    glfwTerminate();
#endif
    exit(return_code);
}

//------------------------------
//-------GLFW CALLBACKS---------
//------------------------------

void errorCallback(int error, const char *description) {
    fputs(description, stderr);
}

void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
	if (action == GLFW_PRESS) {
		switch (key) {
		case GLFW_KEY_ESCAPE:
			glfwSetWindowShouldClose(window, GL_TRUE);
			break;
		}
	}
}





enum ControlState { NONE = 0, ROTATE, TRANSLATE };
ControlState mouseState = NONE;
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
	if (action == GLFW_PRESS)
	{
		if (button == GLFW_MOUSE_BUTTON_LEFT)
		{
			mouseState = ROTATE;
		}
		else if (button == GLFW_MOUSE_BUTTON_RIGHT)
		{
			mouseState = TRANSLATE;
		}

	}
	else if (action == GLFW_RELEASE)
	{
		mouseState = NONE;
	}
	//printf("%d\n", mouseState);
}

double lastx = (double)width / 2;
double lasty = (double)height / 2;
void mouseMotionCallback(GLFWwindow* window, double xpos, double ypos)
{
	const double s_r = 0.01;
	const double s_t = 0.01;

	double diffx = xpos - lastx;
	double diffy = ypos - lasty;
	lastx = xpos;
	lasty = ypos;

	if (mouseState == ROTATE)
	{
		//rotate
		x_angle += (float)s_r * diffy;
		y_angle += (float)s_r * diffx;
		//x_angle = (float)s_r * diffy;
		//y_angle = (float)s_r * diffx;
	}
	else if (mouseState == TRANSLATE)
	{
		//translate
		x_trans += (float)(s_t * diffx);
		y_trans += (float)(-s_t * diffy);
	}
}

void mouseWheelCallback(GLFWwindow* window, double xoffset, double yoffset)
{
	const double s_s = 0.02;

	z_trans += -s_s * yoffset;

	//cout << x_trans<<","<<y_trans<<"," <<z_trans << endl;
	cout << cameraPosition.x << "," << cameraPosition.y << "," << cameraPosition.z << endl;
	cout << cameraPosition.length()<<endl;
}


void updateCamera()
{
	//tmp
	glm::mat4 R =  glm::rotate(x_angle, glm::vec3(1.0f, 0.0f, 0.0f))
		* glm::rotate(y_angle, glm::vec3(0.0f, 1.0f, 0.0f));

	glm::mat4 T = glm::translate(glm::vec3(x_trans, y_trans, z_trans));

	glm::vec4 tmp = R * T * glm::vec4(0.0, 0.0, 0.0, 1.0);
	tmp /= tmp.w;
	cameraPosition = glm::vec3(tmp);

	view = glm::translate(glm::vec3(-x_trans, -y_trans, -z_trans)) * glm::transpose(R);

	projection = glm::lookAt(cameraPosition, glm::vec3(0.0, 0.0, 0.0), glm::vec3(0.0, 1.0, 0.0));
	//projection = glm::lookAt(glm::vec3(0.0,0.0,0.0), cameraPosition, glm::vec3(0.0, 1.0, 0.0));

	//projection = glm::frustum(-5, 5, -5, 5, -4, 10);

	//cout << z_trans << endl;
}