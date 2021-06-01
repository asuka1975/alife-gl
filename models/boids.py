import typing as tp

import numpy as np

from OpenGL.GL import *
import glfw

from . import shader

class Boids:
    vert = """
#version 450

struct agent_t {
    vec2 position;
    vec2 velocity;
};

layout (std430, binding = 0) buffer agents {
    agent_t agent[];
};

layout (location = 0) out vec4 vertexColor;
layout (location = 1) out vec2 direction;

void main() {
    gl_Position = vec4(agent[gl_VertexID].position, 0, 1);
    vertexColor = vec4(1);
    direction = normalize(agent[gl_VertexID].velocity);
}
"""

    geom = """
#version 450

layout (points) in;
layout (triangle_strip, max_vertices = 3) out;

layout (location = 0) in vec4 vertexColor[];
layout (location = 1) in vec2 direction[];
layout (location = 0) out vec4 outColor;

float cross2d(vec2 a, vec2 b) 
{
    return a.x * b.y - a.y * b.x;
}

mat2 rot2d(float cs, float sn) {
    return mat2(
        cs, -sn,
        sn, cs
    );
}

void main(void) {
    float cs = dot(direction[0], vec2(1, 0));
    float sn = cross2d(direction[0], vec2(1, 0));
    vec2 position = gl_in[0].gl_Position.xy;
    mat2 rotm = rot2d(cs, sn);

    gl_Position = vec4(rotm * vec2(0, 0.004) + position, 0, 1);
    outColor = vertexColor[0];
    EmitVertex();
    gl_Position = vec4(rotm * vec2(0, -0.004) + position, 0, 1);
    outColor = vertexColor[0];
    EmitVertex();
    gl_Position = vec4(rotm * vec2(0.02, 0) + position, 0, 1);
    outColor = vertexColor[0];
    EmitVertex();
    EndPrimitive();
}    
"""

    frag = """
#version 450 

layout (location = 0) in vec4 fragColor;
layout (location = 0) out vec4 outColor;

void main() {
    outColor = fragColor;
}
"""

    comp = """
#version 450

layout (local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

struct agent_t {
    vec2 position;
    vec2 velocity;
};

layout (std430, binding = 0) buffer agents {
    agent_t agent[];
};

layout (std430, binding = 1) buffer temps {
    agent_t temp[];
};

uniform int num_agents;
uniform float min_velocity;
uniform float max_velocity;
uniform vec2 mouse_position;

const float pi = 3.1415926535;

const float fc = 0.005;
const float fs = 0.5;
const float fa = 0.01;
const float fb = 0.04;
const float fw = 2.8;
const float dc = 0.8;
const float ds = 0.03;
const float da = 0.5;
const float dw = 0.5;
const float ac = pi / 2;
const float as = pi / 2;
const float aa = pi / 2;


void main() {
    uint work_id = gl_WorkGroupID.z * gl_NumWorkGroups.x * gl_NumWorkGroups.y + 
                   gl_WorkGroupID.y * gl_NumWorkGroups.x +
                   gl_WorkGroupID.x;
    uint id = gl_LocalInvocationIndex + work_id * 32 * 32;

    int num_c = 0;
    int num_s = 0;
    int num_a = 0;
    vec2 f_c = vec2(0);
    vec2 f_s = vec2(0);
    vec2 f_a = vec2(0);
    for(int i = 0; i < num_agents; i++) {
        float l = length(agent[i].position - agent[id].position);
        float a = acos(dot(normalize(agent[i].position), normalize(agent[id].position)));
        f_c += l < dc && a < ac ? (num_c++, agent[i].position) : vec2(0);
        f_s += l < ds && a < as ? (num_s++, agent[id].position - agent[i].position) : vec2(0);
        f_a += l < da && a < aa ? (num_a++, agent[i].velocity) : vec2(0);
    }
    f_c = num_c == 0 ? vec2(0) : (f_c / num_c - agent[id].position) * fc;
    f_s *= fs;
    f_a = num_a == 0 ? vec2(0) : (f_a / num_a - agent[id].velocity) * fa;
    float dist_center = length(agent[id].position);
    vec2 f_b = dist_center > 1 ? -agent[id].position * (dist_center - 1) / dist_center * fb : vec2(0);

    vec2 f_w = length(agent[id].position - mouse_position) < dw ? (agent[id].position - mouse_position) * fw : vec2(0);

    temp[id].velocity = agent[id].velocity + f_c + f_s + f_a + f_b + f_w;
    float lv = length(temp[id].velocity);
    vec2 vdirection = normalize(temp[id].velocity);
    temp[id].velocity = lv < min_velocity ? vdirection * min_velocity : (lv > max_velocity ? vdirection * max_velocity : temp[id].velocity);
    temp[id].position = agent[id].position + temp[id].velocity;
}
"""

    copy = """
#version 450

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

struct agent_t {
    vec2 position;
    vec2 velocity;
};

layout (std430, binding = 0) buffer agents {
    agent_t agent[];
};

layout (std430, binding = 1) buffer temps {
    agent_t temp[];
};

void main() {
    uint work_id = gl_WorkGroupID.z * gl_NumWorkGroups.x * gl_NumWorkGroups.y + 
                   gl_WorkGroupID.y * gl_NumWorkGroups.x +
                   gl_WorkGroupID.x;
    uint id = gl_LocalInvocationIndex + work_id * 32 * 32;

    agent[id] = temp[id];
}
"""

    def __init__(self) -> None:
        self.num_agents = 500

    def setup(self, window: tp.Any) -> None:
        min_velocity = 0.005
        max_velocity = 0.03
        self.window_size = glfw.get_window_size(window)
        self.clicked = False

        self.program = shader.Shader()
        self.program.attach_shader(self.vert, GL_VERTEX_SHADER)
        self.program.attach_shader(self.geom, GL_GEOMETRY_SHADER)
        self.program.attach_shader(self.frag, GL_FRAGMENT_SHADER)
        self.program.link()

        agents = np.random.rand(self.num_agents, 2, 2).astype(np.float32) * 2.0 - 1.0
        agents[:, 1, :] = (np.random.rand(self.num_agents, 2).astype(np.float32) * (max_velocity - min_velocity) + min_velocity) * np.random.choice([1, -1], (self.num_agents, 2))
        ssbo = glGenBuffers(2)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[0])
        glBufferData(GL_SHADER_STORAGE_BUFFER, agents.itemsize * np.prod(agents.shape), agents, GL_STATIC_DRAW)
        temporary = np.zeros(shape=(self.num_agents, 2, 2), dtype=np.float32)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[1])
        glBufferData(GL_SHADER_STORAGE_BUFFER, temporary.itemsize * np.prod(temporary.shape), temporary, GL_STATIC_DRAW)
        
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo[0])
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo[1])

        self.compute = shader.Shader()
        self.compute.attach_shader(self.comp, GL_COMPUTE_SHADER)
        self.compute.link()

        self.memory_copy = shader.Shader()
        self.memory_copy.attach_shader(self.copy, GL_COMPUTE_SHADER)
        self.memory_copy.link()

        self.compute.use()
        glUniform1i(glGetUniformLocation(self.compute.handle, "num_agents"), self.num_agents)
        glUniform1f(glGetUniformLocation(self.compute.handle, "min_velocity"), min_velocity)
        glUniform1f(glGetUniformLocation(self.compute.handle, "max_velocity"), max_velocity)
        self.mouse_position_loc = glGetUniformLocation(self.compute.handle, "mouse_position")
        glUniform2f(self.mouse_position_loc, 500, 500)
        self.compute.unuse()

        glfw.set_cursor_pos_callback(window, self.mouse_pos)
        glfw.set_mouse_button_callback(window, self.mouse_click)

    def render(self) -> None:
        self.compute.use()
        glDispatchCompute(32, 1, 1)
        self.memory_copy.use()
        glDispatchCompute(32, 1, 1)

        self.program.use()
        glDrawArrays(GL_POINTS, 0, self.num_agents)
        self.program.unuse()

    def mouse_pos(self, window, xpos, ypos):
        x = float(xpos) / self.window_size[0]
        y = float(ypos) / self.window_size[1]
        x = x * 2. - 1.
        y = y * 2. - 1.
        y = -y
        self.compute.use()
        if self.clicked:
            glUniform2f(self.mouse_position_loc, x, y)
        self.compute.unuse()

    def mouse_click(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                self.clicked = True
            else:
                self.compute.use()
                glUniform2f(self.mouse_position_loc, 500, 500)
                self.compute.unuse()
                self.clicked = False

