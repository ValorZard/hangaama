use winit::{
    event::*,
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::WindowBuilder,
    window::Window,
};

// need to import this to use create_buffer_init
use wgpu::util::DeviceExt;

#[cfg(target_arch="wasm32")]
use wasm_bindgen::prelude::*;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

const VERTICES: &[Vertex] = &[
    Vertex {position:[0.0, 0.5, 0.0], color: [1.0, 0.0, 0.0]},
    Vertex {position:[-0.5, -0.5, 0.0], color: [0.0, 1.0, 0.0]},
    Vertex {position:[0.0, -0.5, 0.0], color: [0.0, 0.0, 1.0]},
];

struct State<'a> {
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    // window must be declared after surface so it gets dropped after it
    // surface contains unsafe references to window's references
    window: &'a Window,
}

impl<'a> State<'a> {
    // need some async code to create some of the wgpu types
    async fn new(window: &'a Window) -> State<'a> {
        let size = window.inner_size();

        // instance is a handle to our GPU
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            #[cfg(not(target_arch="wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            #[cfg(target_arch="wasm32")]
            backends: wgpu::Backends::GL,
            ..Default::default()
        });

        // part of the window we actually draw to
        let surface = instance.create_surface(window).unwrap();

        // handle for our actual graphics card
        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                // how much power/battery life we will take up
                power_preference: wgpu::PowerPreference::default(),
                // find an adapter that can present to suppiled surface
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            },).await.unwrap();
        
        let (device, queue) = adapter.request_device(
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
        None,).await.unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        // learn wgpu assumes we're using sRGB. I don't know what that is yet, but shouldn't matter for now
        let surface_format = surface_caps.formats.iter()
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

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor { 
            label: Some("Render Pipeline"), 
            layout: Some(&render_pipeline_layout), 
            vertex: wgpu::VertexState { 
                module: &shader, 
                entry_point: "vs_main", 
                // specify what type of vertices we want to pass to the vertex shader
                buffers: &[],
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
                // both also require features we dont have
                unclipped_depth: false,
                conservative: false
             }, 
             // we are not CURRENTLY using a depth/stencil buffer
             depth_stencil: None, 
             multisample: wgpu::MultisampleState {
                count: 1, // how many samples the pipeline will use
                mask: !0, // we want to use all samples
                alpha_to_coverage_enabled: false, // anti-aliasing
             }, 
             // how many array layers render attachments have
             // we aren't using that, so set to none
             multiview: None, 
             // cache shader compilation data (apparently for Android)
             cache: None,
            });

        let vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(VERTICES),
                usage: wgpu::BufferUsages::VERTEX,
            }
        );

        Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            vertex_buffer,
        }

    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>){
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        // dont have any inputs we want to capture (FOR NOW)
        false
    }

    fn update(&mut self) {
        // will put stuff here later
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        // wait for surface to provide SurfaceTexture
        let output = self.surface.get_current_texture()?;
        // use this to control how render code interacts with the texture
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        // create CommandEncoder to create actual commands to send to GPU
        // commands have to be stored in a command buffer to send to GPU
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

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
                    }
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            
            // set pipeline using the one we created
            render_pass.set_pipeline(&self.render_pipeline);
            // draw SOMETHING with three vertices and one instance
            // (this is where @builtin(vertex_index) comes in)
            render_pass.draw(0..3, 0..1);
        }

        // submit will accept anything that implements IntoIter
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}


#[cfg_attr(target_arch="wasm32", wasm_bindgen(start))]
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

    let mut state = State::new(&window).await;
    let mut surface_configured = false;

    event_loop
        .run(move |event, control_flow| {
            match event {
                Event::WindowEvent {
                    ref event,
                    window_id,
                } if window_id == state.window().id() => {
                    if !state.input(event) {
                        // UPDATED!
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
                                log::info!("physical_size: {physical_size:?}");
                                surface_configured = true;
                                state.resize(*physical_size);
                            }
                            WindowEvent::RedrawRequested => {
                                // This tells winit that we want another frame after this one
                                state.window().request_redraw();

                                if !surface_configured {
                                    return;
                                }

                                state.update();
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

