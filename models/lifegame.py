import numpy as np

from OpenGL.GL import *

from . import shader

class Lifegame:
    vert = """
#version 450

layout (std430, binding = 0) buffer cells {
    int cell[];
};

layout (location = 0) out vec4 vertexColor;

uniform ivec2 rect;

void main() {
    float fid = gl_VertexID;
    float x = mod(fid, rect.x) / rect.x;
    float y = int(fid / rect.x) / float(rect.y);

    gl_Position = vec4(vec2(x, y) * 2 - 1, 0, 1);
    vertexColor = vec4(cell[gl_VertexID]);
}
"""

    geom = """
#version 450

layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

layout (location = 0) in vec4 vertexColor[];
layout (location = 0) out vec4 outColor;

uniform ivec2 rect;

void main(void) {
    gl_Position = gl_in[0].gl_Position;
    outColor = vertexColor[0];
    EmitVertex();
    gl_Position = gl_in[0].gl_Position + vec4(2.0 / rect.x, 0, 0, 0);
    outColor = vertexColor[0];
    EmitVertex();
    gl_Position = gl_in[0].gl_Position + vec4(0, 2.0 / rect.y, 0, 0);
    outColor = vertexColor[0];
    EmitVertex();
    gl_Position = gl_in[0].gl_Position + vec4(2.0 / rect.x, 2.0 / rect.y, 0, 0);
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

layout (std430, binding = 0) buffer cells {
    int cell[];
};

layout (std430, binding = 1) buffer temps {
    int temp[];
};

uniform ivec2 rect;

int get(int x, int y) {
    return cell[x + y * rect.x];
}

int neighbor(int x, int y, int dx, int dy) {
    int _x = x + dx < 0 ? rect.x - 1 : x + dx >= rect.x ? 0 : x + dx;
    int _y = y + dy < 0 ? rect.y - 1 : y + dy >= rect.y ? 0 : y + dy;
    return get(_x, _y);
}

void main() {
    uint work_id = gl_WorkGroupID.z * gl_NumWorkGroups.x * gl_NumWorkGroups.y + 
                   gl_WorkGroupID.y * gl_NumWorkGroups.x +
                   gl_WorkGroupID.x;
    uint id = gl_LocalInvocationIndex + work_id * 32 * 32;

    int x = int(mod(id, rect.x));
    int y = int(id / rect.x);

    int num = 0;
    num += neighbor(x, y, -1, 0);
    num += neighbor(x, y, -1, -1);
    num += neighbor(x, y, 0, -1);
    num += neighbor(x, y, 1, -1);
    num += neighbor(x, y, 1, 0);
    num += neighbor(x, y, 1, 1);
    num += neighbor(x, y, 0, 1);
    num += neighbor(x, y, -1, 1);
    temp[id] = num == 3 ? 1 : num == 2 ? cell[id] : 0;
}
"""

    copy = """
#version 450

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

layout (std430, binding = 0) buffer cells {
    int cell[];
};

layout (std430, binding = 1) buffer temps {
    int temp[];
};

void main() {
    uint work_id = gl_WorkGroupID.z * gl_NumWorkGroups.x * gl_NumWorkGroups.y + 
                   gl_WorkGroupID.y * gl_NumWorkGroups.x +
                   gl_WorkGroupID.x;
    uint id = gl_LocalInvocationIndex + work_id * 32 * 32;

    cell[id] = temp[id];
}
"""

    def __init__(self) -> None:
        self.rect = (100, 100)

    def setup(self, resolution: tuple[int, int]) -> None:
        self.program = shader.Shader()
        self.program.attach_shader(self.vert, GL_VERTEX_SHADER)
        self.program.attach_shader(self.geom, GL_GEOMETRY_SHADER)
        self.program.attach_shader(self.frag, GL_FRAGMENT_SHADER)
        self.program.link()

        self.program.use()
        glUniform2i(glGetUniformLocation(self.program.handle, "rect"), *self.rect)
        glUniform2i(glGetUniformLocation(self.program.handle, "resolution"), *resolution)
        self.program.unuse()

        field = np.random.randint(0, 2, self.rect[0] * self.rect[1])
        ssbo = glGenBuffers(2)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[0])
        glBufferData(GL_SHADER_STORAGE_BUFFER, field.itemsize * field.shape[0], field, GL_STATIC_DRAW)
        temporary = np.zeros(field.shape[0])
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[1])
        glBufferData(GL_SHADER_STORAGE_BUFFER, temporary.itemsize * temporary.shape[0], temporary, GL_STATIC_DRAW)

        self.compute = shader.Shader()
        self.compute.attach_shader(self.comp, GL_COMPUTE_SHADER)
        self.compute.link()

        self.memory_copy = shader.Shader()
        self.memory_copy.attach_shader(self.copy, GL_COMPUTE_SHADER)
        self.memory_copy.link()

        self.compute.use()
        glUniform2i(glGetUniformLocation(self.compute.handle, "rect"), *self.rect)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo[0])
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo[1])
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        self.compute.unuse()

        self.memory_copy.use()
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo[0])
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo[1])
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        self.memory_copy.unuse()

    def render(self) -> None:
        self.compute.use()
        glDispatchCompute(64, 64, 1)
        self.memory_copy.use()
        glDispatchCompute(64, 64, 1)

        self.program.use()
        glDrawArrays(GL_POINTS, 0, self.rect[0] * self.rect[1])
        self.program.unuse()

