#include "display.hpp"

#include <chrono>
#include <iostream>
#include <SDL2/SDL.h>
#include <glad/glad.h>
#include <glm/glm.hpp>

using namespace std::chrono_literals;


namespace controls {
    namespace display {

        Display::Trajectory::Trajectory(glm::fvec4 color, GLuint program)
            : color(color), program(program) {

            glGenVertexArrays(1, &VAO);
            glGenBuffers(1, &VBO);

            glBindVertexArray(VAO);

            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, (void*)0);
            glEnableVertexAttribArray(0);

            color_loc = glGetUniformLocation(program, "col");
        }

        void Display::Trajectory::draw() {
            glUniform4f(color_loc, color.x, color.y, color.z, color.w);

            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertex_buf.size(), vertex_buf.data(), GL_DYNAMIC_DRAW);

            glBindVertexArray(VAO);
            glDrawArrays(GL_LINE_STRIP, 0, num_timesteps);
        }

        Display::Display(
            std::shared_ptr<mppi::MppiController> controller,
            std::shared_ptr<state::StateEstimator> state_estimator)
                : m_controller {std::move(controller)},
                  m_state_estimator {std::move(state_estimator)} {
        }

        SDL_Window* Display::init_sdl2() {
            if (SDL_Init(SDL_INIT_VIDEO) < 0) {
                throw std::runtime_error("Failed to initialize SDL2 library");
            }

            SDL_Window* window = SDL_CreateWindow(
                "MPPI Display",
                SDL_WINDOWPOS_CENTERED,
                SDL_WINDOWPOS_CENTERED,
                width, height,
                SDL_WINDOW_OPENGL
            );

            if (!window) {
                throw std::runtime_error("Failed to create window");
            }

            return window;
        }

        void printProgramLog( GLuint program )
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

        void printShaderLog( GLuint shader )
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

        void debugCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam) {
            std::cerr << message << std::endl;
        }

        void Display::init_gl(SDL_Window* window) {
            SDL_GL_SetAttribute( SDL_GL_CONTEXT_MAJOR_VERSION, 3 );
            SDL_GL_SetAttribute( SDL_GL_CONTEXT_MINOR_VERSION, 3 );
            SDL_GL_SetAttribute( SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE );

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

            m_shader_program = glCreateProgram();

            GLuint vertexShader = glCreateShader( GL_VERTEX_SHADER );

            const GLchar* vertexShaderSource[] = {
                "#version 330 core\n"
                "layout (location = 0) in vec2 aPos;\n"
                "uniform vec2 camPos;\n"
                "uniform float camScale;\n"
                "void main()\n"
                "{\n"
                "   gl_Position = vec4((aPos - camPos) / camScale, 0.0f, 1.0f);\n"
                "}"
            };

            glShaderSource(vertexShader, 1, vertexShaderSource, nullptr);
            glCompileShader(vertexShader);

            GLint vShaderCompiled = GL_FALSE;
            glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &vShaderCompiled);
            if(vShaderCompiled != GL_TRUE) {
                printShaderLog(vertexShader);
                throw std::runtime_error("Unable to compile vertex shader");
            }

            GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

            const GLchar* fragmentShaderSource[] = {
                "#version 330 core\n"
                "out vec4 FragColor;\n"
                "uniform vec4 col;"
                "void main()\n"
                "{\n"
                "   FragColor = col;\n"
                "}"
            };

            glShaderSource(fragmentShader, 1, fragmentShaderSource, nullptr);
            glCompileShader(fragmentShader);

            GLint fShaderCompiled = GL_FALSE;
            glGetShaderiv( fragmentShader, GL_COMPILE_STATUS, &fShaderCompiled );
            if( fShaderCompiled != GL_TRUE ) {
                printShaderLog(fragmentShader);
                throw std::runtime_error("Unable to compile fragment shader");
            }

            glAttachShader(m_shader_program, vertexShader);
            glAttachShader(m_shader_program, fragmentShader);
            glLinkProgram(m_shader_program);

            GLint vProgramLinked;
            glGetProgramiv(m_shader_program, GL_LINK_STATUS, &vProgramLinked);
            if(!vProgramLinked) {
                printProgramLog(m_shader_program);
                throw std::runtime_error("Failed to link program");
            }

            m_cam_pos_loc = glGetUniformLocation(m_shader_program, "camPos");
            m_cam_scale_loc = glGetUniformLocation(m_shader_program, "camScale");

            glDeleteShader(vertexShader);
            glDeleteShader(fragmentShader);

            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            glLineWidth(1.0f);
            glDisable(GL_DEPTH_TEST);
            glDisable(GL_CULL_FACE);
        }

        void Display::init_trajectories() {
            for (uint32_t i = 0; i < num_samples; i++) {
                m_trajectories.emplace_back(glm::fvec4 {1.0f, 0.0f, 0.0f, 0.0f}, m_shader_program);
            }
        }

        void Display::init_spline() {
            m_spline = std::make_unique<Trajectory>(glm::fvec4 {1.0f, 1.0f, 1.0f, 1.0f}, m_shader_program);
        }

        void Display::fill_trajectories() {
            using namespace glm;

            std::vector<float> states = m_controller->last_state_trajectories();

            for (uint32_t i = 0; i < num_samples; i++) {
                if (m_trajectories[i].vertex_buf.size() < num_timesteps) {
                    m_trajectories[i].vertex_buf = std::vector<float> (num_samples * 2);
                }

                for (uint32_t j = 0; j < num_timesteps; j++) {
                    uint32_t state_idx = i * state_dims * num_timesteps + j * state_dims;
                    m_trajectories[i].vertex_buf[2 * j] = states[state_idx];
                    m_trajectories[i].vertex_buf[2 * j + 1] = states[state_idx + 1];
                }
            }
        }

        void Display::draw_trajectories() {
            glUseProgram(m_shader_program);
            for (uint32_t i = 0; i < num_samples; i++) {
                Trajectory& t = m_trajectories[i];
                t.draw();
            }
        }

        void Display::draw_spline() {
            auto frames = m_state_estimator->get_spline_frames();

            m_spline->vertex_buf = std::vector<float>(frames.size() * 2);
            for (size_t i = 0; i < frames.size(); i++) {
                m_spline->vertex_buf[2 * i] = frames[i].x;
                m_spline->vertex_buf[2 * i + 1] = frames[i].y;
            }

            m_spline->draw();
        }

        void Display::run() {
            SDL_Window* window = init_sdl2();
            init_gl(window);
            init_trajectories();
            init_spline();

            update_loop(window);
        }

        void Display::update_loop(SDL_Window* window) {
            bool should_close = false;
            auto last_time = std::chrono::system_clock::now();
            float delta_time = 1 / framerate;
            while (!should_close) {
                SDL_Event e;
                while (SDL_PollEvent(&e) > 0) {
                    switch (e.type) {
                        case SDL_QUIT:
                            should_close = true;
                            break;

                        default:
                            break;
                    }
                }

                const uint8_t* keyboard_state = SDL_GetKeyboardState(nullptr);

                if (keyboard_state[SDL_SCANCODE_LEFT]) {
                    m_cam_pos += m_cam_scale * strafe_speed * delta_time * glm::fvec2(-1,0);
                }
                if (keyboard_state[SDL_SCANCODE_RIGHT]) {
                    m_cam_pos += m_cam_scale * strafe_speed * delta_time * glm::fvec2(1,0);
                }
                if (keyboard_state[SDL_SCANCODE_UP]) {
                    m_cam_pos += m_cam_scale * strafe_speed * delta_time * glm::fvec2(0,1);
                }
                if (keyboard_state[SDL_SCANCODE_DOWN]) {
                    m_cam_pos += m_cam_scale * strafe_speed * delta_time * glm::fvec2(0,-1);
                }

                if (keyboard_state[SDL_SCANCODE_S]) {
                    m_cam_scale *= pow(1 + scale_speed, delta_time);
                }
                if (keyboard_state[SDL_SCANCODE_W]) {
                    m_cam_scale *= pow(1 / (1 + scale_speed), delta_time);
                }

                glUseProgram(m_shader_program);
                glUniform2f(m_cam_pos_loc, m_cam_pos.x, m_cam_pos.y);
                glUniform1f(m_cam_scale_loc, m_cam_scale);

                glClear(GL_COLOR_BUFFER_BIT);

                fill_trajectories();
                draw_trajectories();

                draw_spline();

                SDL_GL_SwapWindow(window);

                auto now = std::chrono::system_clock::now();
                uint32_t to_go_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time).count();
                SDL_Delay(std::max(0u, to_go_ms));

                now = std::chrono::system_clock::now();
                delta_time = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time).count() / 1000.0f;
                last_time = now;
            }

            SDL_Quit();
        }

    }
}
