#include <stdexcept>
#include <iostream>
#include <csignal>
#include <SDL2/SDL.h>
#include <glad/glad.h>


constexpr int max_width = 512;
constexpr int max_height = 512;


extern void cuda_test(GLuint rbo, uint width, uint height);


void debugCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam) {
    std::cerr << "\nGL DEBUG: \n" << message << "\n" << std::endl;
}

SDL_Window* init_sdl2_gl(const char* title, int x, int y, int width, int height, Uint32 flags = 0) {
    if (SDL_InitSubSystem(SDL_INIT_VIDEO) < 0) {
        throw std::runtime_error("Failed to initialize SDL2 library");
    }

    SDL_Window* window = SDL_CreateWindow(title, x, y, width, height, SDL_WINDOW_OPENGL | flags);

    if (!window) {
        throw std::runtime_error("Failed to create window");
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_DEBUG_FLAG);
    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    if (!gl_context) {
        std::cerr << SDL_GetError() << std::endl;
        throw std::runtime_error("Failed to create GL context");
    }

    if (!gladLoadGLLoader(SDL_GL_GetProcAddress)) {
        throw std::runtime_error("Failed to initialize GLAD");
    }

    if( SDL_GL_SetSwapInterval(1) < 0 ) {
        throw std::runtime_error("Failed to set vsync");
    }

    glDebugMessageCallbackARB(debugCallback, nullptr);

    return window;
}

void print_program_log(GLuint program)
{
    //Make sure name is shader
    if( glIsProgram( program ) )
    {
        //Program log length
        int infoLogLength = 0;
        int maxLength = infoLogLength;

        //Get info string length
        glGetProgramiv( program, GL_INFO_LOG_LENGTH, &maxLength );

        //Allocate string
        char* infoLog = new char[ maxLength ];

        //Get info log
        glGetProgramInfoLog( program, maxLength, &infoLogLength, infoLog );
        if( infoLogLength > 0 )
        {
            //Print Log
            printf( "%s\n", infoLog );
        }

        //Deallocate string
        delete[] infoLog;
    }
    else
    {
        printf( "Name %d is not a program\n", program );
    }
}

void print_shader_log(GLuint shader)
{
    //Make sure name is shader
    if( glIsShader( shader ) )
    {
        //Shader log length
        int infoLogLength = 0;
        int maxLength = infoLogLength;

        //Get info string length
        glGetShaderiv( shader, GL_INFO_LOG_LENGTH, &maxLength );

        //Allocate string
        char* infoLog = new char[ maxLength ];

        //Get info log
        glGetShaderInfoLog( shader, maxLength, &infoLogLength, infoLog );
        if( infoLogLength > 0 )
        {
            //Print Log
            printf( "%s\n", infoLog );
        }

        //Deallocate string
        delete[] infoLog;
    }
    else
    {
        printf( "Name %d is not a shader\n", shader );
    }
}

GLuint compile_shader(const char *vertex_source, const char *fragment_source) {
    GLuint program_id = glCreateProgram();
    GLuint vertexShader = glCreateShader( GL_VERTEX_SHADER );

    glShaderSource(vertexShader, 1, &vertex_source, nullptr);
    glCompileShader(vertexShader);

    GLint vShaderCompiled = GL_FALSE;
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &vShaderCompiled);
    if(vShaderCompiled != GL_TRUE) {
        print_shader_log(vertexShader);
        throw std::runtime_error("Unable to compile vertex shader");
    }

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

    glShaderSource(fragmentShader, 1, &fragment_source, nullptr);
    glCompileShader(fragmentShader);

    GLint fShaderCompiled = GL_FALSE;
    glGetShaderiv( fragmentShader, GL_COMPILE_STATUS, &fShaderCompiled );
    if( fShaderCompiled != GL_TRUE ) {
        print_shader_log(fragmentShader);
        throw std::runtime_error("Unable to compile fragment shader");
    }

    glAttachShader(program_id, vertexShader);
    glAttachShader(program_id, fragmentShader);
    glLinkProgram(program_id);

    GLint vProgramLinked;
    glGetProgramiv(program_id, GL_LINK_STATUS, &vProgramLinked);
    if(!vProgramLinked) {
        print_program_log(program_id);
        throw std::runtime_error("Failed to link program");
    }

    return program_id;
}

GLuint create_triangle() {
    GLuint VAO, VBO;

    float vertices[] = {
        -0.5f, -0.5f, 0.0f, // left
        0.5f, -0.5f, 0.0f, // right
        0.0f,  0.5f, 0.0f  // top
    };

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);

    return VAO;
}

void draw_triangles(GLuint shader, GLuint VAO, GLsizei count) {
    glUseProgram(shader);
    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, count * 3);
}

void  gen_framebuffer(GLsizei width, GLsizei height, GLuint& fbo, GLuint& rbo) {
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA32F, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rbo);

    //add a depth buffer
    GLuint depth_rbo;
    glGenRenderbuffers(1, &depth_rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, depth_rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_rbo);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        throw std::runtime_error("Framebuffer is not complete");
    }
}

int main() {
    SDL_Window* window = init_sdl2_gl("gl experiment", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, max_width, max_height, 0);

    const char *vertex_source = R"(
        #version 330 core
        layout (location = 0) in vec3 aPos;
        void main() {
            gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);
        }
    )";

    const char *fragment_source = R"(
        #version 330 core
        out vec4 FragColor;
        void main() {
            FragColor = vec4(1.0f, 1.0f, 0.0f, 1.0f);
        }
    )";

    GLuint fbo, rbo;
    gen_framebuffer(max_width, max_height, fbo, rbo);

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    GLuint shader_program = compile_shader(vertex_source, fragment_source);
    GLuint triangle_vao = create_triangle();

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    draw_triangles(shader_program, triangle_vao, 1);

    cuda_test(rbo, max_width, max_height);

    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glBlitFramebuffer(0, 0, max_width, max_height, 0, 0, max_width, max_height, GL_COLOR_BUFFER_BIT, GL_NEAREST);

    SDL_GL_SwapWindow(window);

    sigset_t set;
    sigfillset(&set);
    sigdelset(&set, SIGINT);
    sigdelset(&set, SIGTERM);
    sigdelset(&set, SIGQUIT);
    sigsuspend(&set);

    return 0;
}