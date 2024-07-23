use winit::{
    event::*,
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::WindowBuilder,
    window::Window,
};

#[cfg(target_arch="wasm32")]
use wasm_bindgen::prelude::*;

struct State<'a> {
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
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

        Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
        }

    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>){
        todo!()
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        todo!()
    }

    fn update(&mut self) {
        todo!()
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        todo!()
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

    event_loop.run(move |event, control_flow| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == state.window().id() => match event {
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
            _ => {}
        },
        _ => {}
    });

    
}

