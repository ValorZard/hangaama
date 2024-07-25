pub mod texture;

pub struct Sprite {
    texture: texture::Texture,
    pub bind_group: wgpu::BindGroup,
}

impl Sprite {
    pub fn new(
        image_path: &str,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
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

        Sprite {
            texture,
            bind_group,
        }
    }
}
