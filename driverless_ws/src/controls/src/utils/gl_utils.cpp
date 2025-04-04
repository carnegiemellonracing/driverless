#include "gl_utils.hpp"

#include <iostream>
#include <stdexcept>
#include <glad/glad.h>
#include <utils/general_utils.hpp>


namespace controls {
    namespace utils {
        SDL_Window* create_sdl2_gl_window(const char *title, int width, int height, Uint32 additional_flags, SDL_GLContext* gl_context_out) {
            if (SDL_InitSubSystem(SDL_INIT_VIDEO) < 0) {
                throw std::runtime_error("Failed to initialize SDL2 library");
            }

            SDL_Window* window = SDL_CreateWindow(
                title,
                SDL_WINDOWPOS_CENTERED,
                SDL_WINDOWPOS_CENTERED,
                width, height,
                SDL_WINDOW_OPENGL | additional_flags
            );

            if (!window) {
                throw std::runtime_error("Failed to create window");
            }

            SDL_GL_SetAttribute( SDL_GL_CONTEXT_MAJOR_VERSION, 3 );
            SDL_GL_SetAttribute( SDL_GL_CONTEXT_MINOR_VERSION, 3 );
            SDL_GL_SetAttribute( SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE );

            // get verbose gl warnings
            // SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_DEBUG_FLAG);

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

            // glDebugMessageCallbackARB(gl_debug_callback, nullptr);

            if (gl_context_out != nullptr) {
                *gl_context_out = gl_context;
            }

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

        void gl_debug_callback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam) {
            std::cerr << message << std::endl;
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
                throw ControllerError("Failed to link program");
            }

            return program_id;
        }

        void make_gl_current_or_except(SDL_Window* window, SDL_GLContext gl_context) {
            if (SDL_GL_MakeCurrent(window, gl_context) < 0) {
                throw ControllerError(SDL_GetError());
            }
        }

        void sync_gl_and_unbind_context(SDL_Window *window) {
            glFinish();
            make_gl_current_or_except(window, nullptr);
        }

    }
}
