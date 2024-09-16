use wgpu::util::DeviceExt;

use crate::Vertex;

pub mod texture;

pub struct Sprite {
    texture: texture::Texture,
    pub bind_group: wgpu::BindGroup,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_indices: u32,
}

impl Sprite {
    pub fn new(
        image_path: &str,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let _span = tracy_client::span!("Sprite::new()");
        // grab image from file
        let diffuse_bytes = std::fs::read(image_path).unwrap();
        let texture =
            texture::Texture::from_bytes(device, queue, &diffuse_bytes, image_path).unwrap();

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&texture.sampler),
                },
            ],
            label: Some("diffuse_bind_group"),
        });

        // convert textures to texture scale with regards to size of screen

        // TODO: Unhardcode this
        let SCALE_FACTOR : f32 = 100.;

        let screen_width : f32 = texture.width as f32 / SCALE_FACTOR;
        let screen_height : f32 = texture.height as f32 / SCALE_FACTOR;

        // store all UNIQUE vertices
        // this saves a lot of memory compared to storing every single vertex
        let VERTICES: &[Vertex] = &[
            // Changed
            Vertex {
                position: [screen_width / 2., screen_height / 2., 0.0],
                tex_coords: [1.0, 0.0],
            }, // A
            Vertex {
                position: [-screen_width / 2., screen_height / 2., 0.0],
                tex_coords: [0.0, 0.0],
            }, // B
            Vertex {
                position: [-screen_width / 2., -screen_height / 2., 0.0],
                tex_coords: [0.0, 1.0],
            }, // C
            Vertex {
                position: [screen_width / 2., -screen_height / 2., 0.0],
                tex_coords: [1.0, 1.0],
            },
        ];

        // we can reuse vertices to create the triangles
        const INDICES: &[u16] = &[0, 1, 2, 2, 3, 0];

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });

        let num_indices = INDICES.len() as u32;

        Sprite {
            texture,
            bind_group,
            vertex_buffer,
            index_buffer,
            num_indices,
        }
    }
}
