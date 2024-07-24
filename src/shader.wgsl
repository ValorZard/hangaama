// vertex shader
struct CameraUniform {
    view_proj: mat4x4<f32>,
};

// number is determined by render_pipeline_layout
// texture_bind_group_layout is listed first (so group(0))
// camera_bind_group is second, so it's group(1)
@group(1) @binding(0)
var<uniform> camera: CameraUniform;

struct VertextInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
};

struct InstanceInput {
    @location(5) model_matrix_0: vec4<f32>,
    @location(6) model_matrix_1: vec4<f32>,
    @location(7) model_matrix_2: vec4<f32>,
    @location(8) model_matrix_3: vec4<f32>,
};


@vertex
fn vs_main(
    model: VertextInput,
    instance: InstanceInput,
) -> VertexOutput {
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );
    var out: VertexOutput;
    out.tex_coords = model.tex_coords;
    // multiplication order is important 
    // vector goes on the right, and matrices go on the left in order of importance
    out.clip_position = camera.view_proj * model_matrix * vec4<f32>(model.position, 1.0);
    return out;
}

// fragment shader

// these are uniforms
@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // set color
    return textureSample(t_diffuse, s_diffuse, in.tex_coords);
}