mod sprite;
use std::{collections::HashMap, time::Instant};
use glyphon::{
    Attrs, Buffer, Cache, Color, Family, FontSystem, Metrics, Resolution, Shaping, SwashCache,
    TextArea, TextAtlas, TextBounds, TextRenderer, Viewport,
};
use log::Log;
use crate::sprite::*;
mod camera;
use camera::*;
mod input;
use input::InputStruct;
use winit::{
    event::*,
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
    window::WindowBuilder,
};
use rand::prelude::*;

// need to import this to use create_buffer_init
use cgmath::{prelude::*, Vector2};
use wgpu::{util::DeviceExt, MultisampleState};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    tex_coords: [f32; 2], // use colors from texture
}

impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            // how wide a vertex is (sahder will skip this amount of bytes)
            array_stride: mem::size_of::<Vertex>() as wgpu::BufferAddress,
            // each element of the array in this buffer represents per-vertex data (for now)
            step_mode: wgpu::VertexStepMode::Vertex,
            // describe individual parts of the vertex
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0, // offset is in bytes, first attribute is usually zero
                    // corresponds with @location in the shader file
                    shader_location: 0,
                    // shape of the attribute (max size is Float32x4)
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    // later offsets of attributes is the sum over size_of previous attributes data
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

struct Instance {
    position: cgmath::Vector3<f32>,
    rotation: cgmath::Quaternion<f32>,
    scaling: cgmath::Vector3<f32>,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceRaw {
    model: [[f32; 4]; 4],
}

impl Instance {
    fn to_raw(&self) -> InstanceRaw {
        InstanceRaw {
            model: (cgmath::Matrix4::from_translation(self.position)
                * cgmath::Matrix4::from(self.rotation) * cgmath::Matrix4::from_nonuniform_scale(self.scaling.x, self.scaling.y, self.scaling.z))
            .into(),
        }
    }
}

impl InstanceRaw {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            // We need to switch from using a step mode of Vertex to Instance
            // This means that our shaders will only change to use the next
            // instance when the shader starts processing a new instance
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                // A mat4 takes up 4 vertex slots as it is technically 4 vec4s. We need to define a slot
                // for each vec4. We'll have to reassemble the mat4 in the shader.
                wgpu::VertexAttribute {
                    offset: 0,
                    // While our vertex shader only uses locations 0, and 1 now, in later tutorials, we'll
                    // be using 2, 3, and 4, for Vertex. We'll start at slot 5, not conflict with them later
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

const NUM_INSTANCES_PER_ROW: u32 = 10;
const INSTANCE_DISPLACEMENT: cgmath::Vector3<f32> = cgmath::Vector3::new(
    NUM_INSTANCES_PER_ROW as f32 * 0.5,
    0.0,
    NUM_INSTANCES_PER_ROW as f32 * 0.5,
);
struct RenderBlock {
    sprite: Sprite,
    instances: Vec<Instance>,
}

impl RenderBlock {
    fn new(
        image_path: &str,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let _span = tracy_client::span!("RenderBlock::new()");
        let sprite = Sprite::new(image_path, device, queue, texture_bind_group_layout);
        // loop through and make a square of texture instances
        let instances = Vec::new();

        Self {
            sprite,
            instances,
        }
    }

    // make sure you can't have scale be negative or things will break
    fn add_instance(&mut self, x: f32, y: f32, rotation: f32, scale_x: f32, scale_y: f32){
        let position = cgmath::Vector3 {
            x: x,
            y: y,
            z: 0.0,
        };

        let rotation = cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Deg(rotation));

        let scaling = cgmath::Vector3 {
            x: scale_x,
            y: scale_y,
            z: 1.0,
        };

        let instance = Instance { position, rotation, scaling };

        let _ = &self.instances.push(instance);
    }

    fn clear_instances(&mut self){
        let _ = &self.instances.clear();
    }

    fn get_instance_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer
    {
        // refresh these bits
        let instance_data = &self.instances.iter().map(Instance::to_raw).collect::<Vec<_>>();

        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(instance_data),
            usage: wgpu::BufferUsages::VERTEX,
        })
    }
}

struct RenderState<'a> {
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    // each render block is tied to the string path of where the asset is from
    asset_map: std::collections::HashMap<&'static str, RenderBlock>,
    camera: Camera,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    camera_controller: CameraController,
    input_struct: InputStruct,
    // text stuff
    font_system: FontSystem,
    swash_cache: SwashCache,
    viewport: glyphon::Viewport,
    atlas: glyphon::TextAtlas,
    text_renderer: glyphon::TextRenderer,
    text_buffers: Vec<glyphon::Buffer>,

    // window must be declared after surface so it gets dropped after it
    // surface contains unsafe references to window's references
    window: &'a Window,
}

impl<'a> RenderState<'a> {
    // need some async code to create some of the wgpu types
    async fn new(window: &'a Window) -> RenderState<'a> {
        let size = window.inner_size();
        // instance is a handle to our GPU
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::GL,
            ..Default::default()
        });

        // part of the window we actually draw to
        let surface = instance.create_surface(window).unwrap();

        // handle for our actual graphics card
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                // how much power/battery life we will take up
                power_preference: wgpu::PowerPreference::default(),
                // find an adapter that can present to supplied surface
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    // disable features in wgpu that wont work with WebGL
                    // sidenote, WebGL sucks ass
                    required_limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                    label: None,
                    memory_hints: Default::default(), // guess they forgot this
                },
                None,
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        // learn wgpu assumes we're using sRGB. I don't know what that is yet, but shouldn't matter for now
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            // we are going to use these textures to write to the screen
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            // how SurfaceTexture's will be stored on the GPU
            format: surface_format,
            // width and height in pixels of the SurfaceTexture
            // DON'T MAKE THEM ZERO, WILL MAKE IT CRASH!!!!
            width: size.width,
            height: size.height,
            // we use default (this is an enum that controls if VSync is on or off)
            present_mode: surface_caps.present_modes[0],
            // even the learn wgpu guy doesn't know what this is
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    // sampled texture at binding 0
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    // sampled texture at binding 1
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        // this should match the filterable field of the corresponding Texture entry above
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        // we should refactor this at some point

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        // do camera stuff
        let camera = Camera {
            // position the camera 1 unit up and 2 units back
            // +z is out of the screen
            eye: (0.0, 0.0, 1.0).into(),
            // have it look at the origin
            target: (0.0, 0.0, 0.0).into(),
            // which way is "up"
            up: cgmath::Vector3::unit_y(),
            aspect: config.width as f32 / config.height as f32,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
        };

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    // only need vertex data for the camera
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        // we expect the location of the data won't change
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("camera_bind_group_layout"),
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        let camera_controller = CameraController::new(0.2);

        // render pipeline
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&texture_bind_group_layout, &camera_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                // specify what type of vertices we want to pass to the vertex shader
                buffers: &[Vertex::desc(), InstanceRaw::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            // technically optional, stores the color we want
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                // what color outputs wgpu should set up
                // currently: only need one for the surface
                // will come back to when we start doing textures
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                // ever three vertices will correspond to one triangle
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                // Ccw means a triangle is facing forward if vertices are arranged counter-clockwise
                front_face: wgpu::FrontFace::Ccw,
                // Triangle that are not facing forward get culled (CullMode::Back)
                cull_mode: Some(wgpu::Face::Back),
                // kinda HAS to be fill
                polygon_mode: wgpu::PolygonMode::Fill,
                // both also require features we don't have
                unclipped_depth: false,
                conservative: false,
            },
            // we are not CURRENTLY using a depth/stencil buffer
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,                         // how many samples the pipeline will use
                mask: !0,                         // we want to use all samples
                alpha_to_coverage_enabled: false, // anti-aliasing
            },
            // how many array layers render attachments have
            // we aren't using that, so set to none
            multiview: None,
            // cache shader compilation data (apparently for Android)
            cache: None,
        });

        
        // Set up text renderer
        let font_system = FontSystem::new();
        let swash_cache = SwashCache::new();
        let cache = Cache::new(&device);
        let viewport = Viewport::new(&device, &cache);
        let mut atlas = TextAtlas::new(&device, &queue, &cache, surface_format);
        let text_renderer =
            TextRenderer::new(&mut atlas, &device, MultisampleState::default(), None);

        let text_buffers = Vec::<glyphon::Buffer>::new();

        Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            texture_bind_group_layout,
            asset_map: HashMap::new(),
            camera,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            camera_controller,
            input_struct: InputStruct::new(),
            font_system,
            swash_cache,
            viewport,
            atlas,
            text_renderer,
            text_buffers,
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        self.input_struct.process_events(event)
    }

    // image path has to be a string literal
    fn load_asset(&mut self, image_path: &'static str)
    {
        let _span = tracy_client::span!("add render block");
        if self.asset_map.contains_key(image_path) {
            return;
        }
        let render_block = RenderBlock::new(
            image_path,
            &self.device,
            &self.queue,
            &self.texture_bind_group_layout,
        );
        self.asset_map.insert(image_path, render_block);
    }

    fn delete_asset(&mut self, image_path: &'static str){
        self.asset_map.remove(image_path);
    }

    fn add_render_instance(&mut self, image_path: &'static str, x: f32, y: f32){
        self.add_render_instance_with_rotation_and_scaling(image_path, x, y, 0.0, 1.0, 1.0);
    }

    fn add_render_instance_with_rotation(&mut self, image_path: &'static str, x: f32, y: f32, rotation: f32){
        self.add_render_instance_with_rotation_and_scaling(image_path, x, y, rotation, 1.0, 1.0);
    }

    fn add_render_instance_with_scaling(&mut self, image_path: &'static str, x: f32, y: f32, scale_x: f32, scale_y: f32){
        self.add_render_instance_with_rotation_and_scaling(image_path, x, y, 0.0, scale_x, scale_y);
    }

    fn add_render_instance_with_rotation_and_scaling(&mut self, image_path: &'static str, x: f32, y: f32, rotation: f32, scale_x: f32, scale_y: f32){
        let _span = tracy_client::span!("add render instance");
        // add asset if it doesn't already exist
        if !self.asset_map.contains_key(image_path) {
            self.load_asset(image_path);
        }
        self.asset_map.get_mut(image_path).unwrap().add_instance(x, y, rotation, scale_x, scale_y);
    }

    fn set_text(&mut self, text : &str)
    {
        let size = self.window.inner_size();
        let scale_factor = self.window.scale_factor();

        let mut text_buffer = Buffer::new(&mut self.font_system, Metrics::new(30.0, 42.0));

        let physical_width = (size.width as f64 * scale_factor) as f32;
        let physical_height = (size.height as f64 * scale_factor) as f32;

        text_buffer.set_size(
            &mut self.font_system,
            Some(physical_width),
            Some(physical_height),
        );
        text_buffer.set_text(&mut self.font_system,  text, Attrs::new().family(Family::SansSerif), Shaping::Advanced);
        text_buffer.shape_until_scroll(&mut self.font_system, false);
        self.text_buffers.push(text_buffer);
    }

    fn update(&mut self) {
        let _span = tracy_client::span!("update game logic");
        // usually we should have a seperate buffer called a staging buffer
        // use this second buffer to copy in the contents of our camera_buffer
        // this would let the GPU to do some optimizations
        // but we're not going to do that lol
        self.camera_controller.update_camera(&mut self.camera);
        self.camera_uniform.update_view_proj(&self.camera);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let _span = tracy_client::span!("call render()");
        // wait for surface to provide SurfaceTexture
        let output = self.surface.get_current_texture()?;
        // use this to control how render code interacts with the texture
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.viewport.update(&self.queue, Resolution {
            width: self.size.width,
            height: self.size.height,
        });
        // create CommandEncoder to create actual commands to send to GPU
        // commands have to be stored in a command buffer to send to GPU
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // text stuff

        let mut text_areas = Vec::<TextArea>::new();
        
        let mut spacing = 0;

        for buffer in &self.text_buffers {
            text_areas.push(TextArea {
                buffer,
                left: 10.0,
                top: 10.0 + (spacing as f32),
                scale: 1.0,
                bounds: TextBounds {
                    left: 0,
                    top: 0,
                    right: 600,
                    bottom: 50 + spacing,
                },
                default_color: Color::rgb(255, 255, 255),
            });
            // add spacing
            spacing += 50;
        }

        self.text_renderer
                    .prepare(
                        &self.device,
                        &self.queue,
                        &mut self.font_system,
                        &mut self.atlas,
                        &self.viewport,
                        text_areas,
                        &mut self.swash_cache,
                    )
                    .unwrap();

        
        // clearing the screen using a RenderPass
        // put this inside a block so that we can tell rust to drop all variables inside of it
        // because of borrow checker shenanigans
        // we could have also done drop(render_pass)
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                // this is where we will be drawing our color to
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    // render to screen
                    view: &view,
                    // only important for multisampling
                    resolve_target: None,
                    // tells wgpu what to do with the colors on screen
                    ops: wgpu::Operations {
                        // color we're setting the screen to
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            // set pipeline using the one we created
            render_pass.set_pipeline(&self.render_pipeline);

            // render all "render blocks"
            for render_block in self.asset_map.values_mut() { 
                // use our BindGroup
                render_pass.set_bind_group(0, &render_block.sprite.bind_group, &[]);
                // set camera bind group
                render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
                // have to set vertex buffer in render method, else everything will crash
                render_pass.set_vertex_buffer(0, render_block.sprite.vertex_buffer.slice(..));
                render_pass.set_vertex_buffer(1, render_block.get_instance_buffer(&self.device).slice(..));
                // we can have only one index buffer at a time
                render_pass.set_index_buffer(
                    render_block.sprite.index_buffer.slice(..),
                    wgpu::IndexFormat::Uint16,
                );
                // we're using draw_indexed not draw(), since draw ignores index buffer.
                render_pass.draw_indexed(
                    0..render_block.sprite.num_indices,
                    0,
                    0..render_block.instances.len() as _,
                );
                // since this is being called every frame (i think), clear the instances since we might change them later
                render_block.clear_instances();
            }

            // render text last
            self.text_renderer.render(&self.atlas, &self.viewport, &mut render_pass).unwrap();
        }

        // submit will accept anything that implements IntoIter
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        let _ = &self.atlas.trim();

        // delete text buffers
        self.text_buffers.clear();

        Ok(())
    }

    fn clear_assets(&mut self){
        self.asset_map.clear()
    }

}

const PLAYER_SPEED : f32 = 200.0;
const GRAVITY : f32 = -200.0;
struct Player {
    position: Vector2<f32>,
}

const SPAWN_TIME : f32 = 0.1;
const SPIKE_SPEED : f32 = 100.;
const SPIKE_LIFETIME : f32 = 0.3;

struct Spike {
    position: Vector2<f32>,
    facing_up: bool,
    lifetime: f32,
}

struct LogicState {
    player: Player,
    spikes: Vec<Spike>,
    spawn_timer: f32,
    rng: ThreadRng,
}

impl LogicState {
    pub fn new() -> Self
    {
        Self {
            player: Player {
                position: Vector2::<f32>::new(0.0, 0.0),
            },
            spikes: Vec::<Spike>::new(),
            spawn_timer: 0.,
            rng: rand::thread_rng(),
        }
    }
    pub fn update(&mut self, input : &InputStruct, delta_time: f32) {

        let mut velocity_x = 0.0;
        let mut velocity_y = GRAVITY * delta_time;

        if input.is_space_pressed {
            velocity_y = PLAYER_SPEED  * delta_time;
        }


        self.player.position.x += velocity_x;
        self.player.position.y += velocity_y;

        self.player.position.y = if self.player.position.y > 15. {
            15.
        } else if self.player.position.y < -15. {
            -15.
        }
        else {
            self.player.position.y
        };

        // spike/pipe stuff

        self.spawn_timer += delta_time;

        if self.spawn_timer > SPAWN_TIME {
            let offset = self.rng.gen_range(-10.0..=10.0);
            self.spikes.push(Spike { position : Vector2::<f32>::new(10., -15. + offset), facing_up: true, lifetime: SPIKE_LIFETIME});
            self.spikes.push(Spike { position : Vector2::<f32>::new(10., 15. + offset), facing_up: false, lifetime: SPIKE_LIFETIME});
            self.spawn_timer = 0.;
        }

        for spike in &mut self.spikes {
            spike.position.x -= SPIKE_SPEED * delta_time;
            spike.lifetime -= delta_time;
        }
        
        self.spikes.retain(|spike| spike.lifetime > 0.);

    }
}

fn game_logic(input: &InputStruct, logic: &mut LogicState, delta_time: f32){
    // player controller
    logic.update(input, delta_time);
}

fn game_render(state: &mut RenderState, logic: &mut LogicState){
    // this will lag the first time this is called since we're loading it in for the first time
    state.add_render_instance_with_scaling("src/yellowbird-downflap.png", logic.player.position.x, logic.player.position.y, 10., 10.);
    for spike in &logic.spikes
    {
        state.add_render_instance_with_rotation_and_scaling("src/pipe-green.png", spike.position.x, spike.position.y, if spike.facing_up {0.} else {180.}, 5., 8.);
    }
    state.add_render_instance_with_scaling("src/happy-tree-cartoon.png", 8.0, 9.0, 2.0, 0.4);
    state.add_render_instance_with_rotation_and_scaling("src/happy-tree-cartoon.png", -5.0, 5.0, 32.0, 1.2, 2.2);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            // we need a different logger for WASM world
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init_with_level(log::Level::Warn).expect("Couldn't initialize logger");
        } else {
            env_logger::init();
        }
    }

    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    #[cfg(target_arch = "wasm32")]
    {
        // Winit prevents sizing with CSS, so we have to set
        // the size manually when on web.
        use winit::dpi::PhysicalSize;
        let _ = window.request_inner_size(PhysicalSize::new(450, 400));

        use winit::platform::web::WindowExtWebSys;
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| {
                let dst = doc.get_element_by_id("wasm-container")?;
                let canvas = web_sys::Element::from(window.canvas()?);
                dst.append_child(&canvas).ok()?;
                Some(())
            })
            .expect("Couldn't append canvas to document body.");
    }

    let mut state = RenderState::new(&window).await;
    let mut logic = LogicState::new();
    let mut surface_configured = false;
    let mut now = Instant::now();

    let _run_span = tracy_client::span!("begin actual run()");

    event_loop
        .run(move |event, control_flow| {
            match event {
                Event::WindowEvent {
                    ref event,
                    window_id,
                } if window_id == state.window().id() => {
                    if !state.input(event) {
                        match event {
                            WindowEvent::CloseRequested
                            | WindowEvent::KeyboardInput {
                                event:
                                    KeyEvent {
                                        state: ElementState::Pressed,
                                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                                        ..
                                    },
                                ..
                            } => control_flow.exit(),
                            WindowEvent::Resized(physical_size) => {
                                surface_configured = true;
                                state.resize(*physical_size);
                            }
                            WindowEvent::RedrawRequested => {
                                // This tells winit that we want another frame after this one
                                state.window().request_redraw();

                                if !surface_configured {
                                    return;
                                }

                                let _span = tracy_client::span!("game loop");

                                // get delta time
                                let delta_time = now.elapsed().as_secs_f32();
                                state.set_text(&format!("FPS: {}", 1.0 / delta_time));
                                state.set_text(&format!("Position: {0}, {1}", logic.player.position.x, logic.player.position.y));
                                state.set_text(&format!("Spikes: {0}", logic.spikes.len()));
                                state.update();

                                game_logic(&state.input_struct, &mut logic, delta_time);
                                
                                let _span2 = tracy_client::span!("start render");
                                
                                game_render(&mut state, &mut logic);
                                
                                // render
                                match state.render() {
                                    Ok(_) => {}
                                    // Reconfigure the surface if it's lost or outdated
                                    Err(
                                        wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated,
                                    ) => state.resize(state.size),
                                    // The system is out of memory, we should probably quit
                                    Err(wgpu::SurfaceError::OutOfMemory) => {
                                        log::error!("OutOfMemory");
                                        control_flow.exit();
                                    }
                                    // This happens when the a frame takes too long to present
                                    Err(wgpu::SurfaceError::Timeout) => {
                                        log::warn!("Surface timeout")
                                    }
                                }

                                // reset deltatime
                               now = Instant::now();
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        })
        .unwrap();
}
