use miniquad::*;

mod vertices0;
mod vertices1;
mod vertices2;

mod indices0;
mod indices1;
mod indices2;

#[derive(Debug)]
#[repr(C)]
pub struct Vertex {
    pub pos: [f32; 2],
    pub uv: [f32; 2],
    pub color: [u8; 4],
}

pub struct Painter {
    pipeline: Pipeline,
    bindings: Bindings,
}

impl Painter {
    pub fn new(ctx: &mut Context) -> Painter {
        let shader = Shader::new(ctx, shader::VERTEX, shader::FRAGMENT, shader::meta());

        let pipeline = Pipeline::with_params(
            ctx,
            &[BufferLayout::default()],
            &[
                VertexAttribute::new("a_pos", VertexFormat::Float2),
                VertexAttribute::new("a_tc", VertexFormat::Float2),
                VertexAttribute::new("a_srgba", VertexFormat::Byte4),
            ],
            shader.expect("couldn't make shader"),
            PipelineParams {
                color_blend: Some(BlendState::new(
                    Equation::Add,
                    BlendFactor::One,
                    BlendFactor::OneMinusValue(BlendValue::SourceAlpha),
                )),
                ..Default::default()
            },
        );

        let vertex_buffer = Buffer::stream(
            ctx,
            BufferType::VertexBuffer,
            32 * 1024 * std::mem::size_of::<Vertex>(),
        );
        let index_buffer = Buffer::stream(
            ctx,
            BufferType::IndexBuffer,
            32 * 1024 * std::mem::size_of::<u16>(),
        );
        let bindings = Bindings {
            vertex_buffers: vec![vertex_buffer],
            index_buffer,
            images: vec![miniquad::Texture::empty()],
        };

        Painter { pipeline, bindings }
    }

    fn rebuild_egui_texture(&mut self, ctx: &mut Context) {
        let texture_data = include_bytes!("texture.png");

        let img = image::load_from_memory(texture_data).unwrap().to_rgba8();
        let width = dbg!(img.width() as u16);
        let height = dbg!(img.height() as u16);
        let bytes = img.into_raw();

        self.bindings.images[0] = miniquad::Texture::from_data_and_format(
            ctx,
            &bytes,
            miniquad::TextureParams {
                format: miniquad::TextureFormat::RGBA8,
                wrap: miniquad::TextureWrap::Clamp,
                filter: miniquad::FilterMode::Linear,
                width: width as _,
                height: height as _,
            },
        );
    }

    fn paint_job(&mut self, ctx: &mut Context, vertices: &[Vertex], indices: &[u16]) {
        let screen_size_in_pixels = ctx.screen_size();
        let pixels_per_point = ctx.dpi_scale();

        let vertices_size_bytes = vertices.len() * std::mem::size_of::<Vertex>();
        if self.bindings.vertex_buffers[0].size() < vertices_size_bytes {
            self.bindings.vertex_buffers[0].delete();
            self.bindings.vertex_buffers[0] =
                Buffer::stream(ctx, BufferType::VertexBuffer, vertices_size_bytes);
        }
        self.bindings.vertex_buffers[0].update(ctx, &vertices);

        let indices_size_bytes = indices.len() * std::mem::size_of::<u16>();
        if self.bindings.index_buffer.size() < indices_size_bytes {
            self.bindings.index_buffer.delete();
            self.bindings.index_buffer =
                Buffer::stream(ctx, BufferType::IndexBuffer, indices_size_bytes);
        }
        self.bindings.index_buffer.update(ctx, &indices);

        let (width_in_pixels, height_in_pixels) = screen_size_in_pixels;

        ctx.apply_bindings(&self.bindings);
        ctx.draw(0, indices.len() as i32, 1);
    }
}

struct Stage {
    painter: Painter,
}

impl Stage {
    fn new(ctx: &mut Context) -> Self {
        let mut painter = Painter::new(ctx);
        painter.rebuild_egui_texture(ctx);
        Self { painter }
    }
}

impl EventHandler for Stage {
    fn update(&mut self, _ctx: &mut Context) {}

    fn draw(&mut self, ctx: &mut Context) {
        ctx.clear(Some((1., 1., 1., 1.)), None, None);
        ctx.begin_default_pass(PassAction::clear_color(0.0, 0.0, 0.0, 1.0));
        ctx.end_render_pass();

        // Draw things in front of egui here

        ctx.begin_default_pass(miniquad::PassAction::Nothing);
        ctx.apply_pipeline(&self.painter.pipeline);

        let screen_size_in_pixels = ctx.screen_size();
        let screen_size_in_points = (
            screen_size_in_pixels.0 / ctx.dpi_scale(),
            screen_size_in_pixels.1 / ctx.dpi_scale(),
        );
        ctx.apply_uniforms(&shader::Uniforms {
            u_screen_size: screen_size_in_points,
        });

        self.painter
            .paint_job(ctx, &vertices0::VERTICES, &indices0::INDICES);
        self.painter
            .paint_job(ctx, &vertices1::VERTICES, &indices1::INDICES);
        self.painter
            .paint_job(ctx, &vertices2::VERTICES, &indices2::INDICES);

        ctx.end_render_pass();

        ctx.commit_frame();
    }

    fn mouse_motion_event(&mut self, ctx: &mut Context, x: f32, y: f32) {}

    fn mouse_wheel_event(&mut self, ctx: &mut Context, dx: f32, dy: f32) {}

    fn mouse_button_down_event(&mut self, ctx: &mut Context, mb: MouseButton, x: f32, y: f32) {}

    fn mouse_button_up_event(&mut self, ctx: &mut Context, mb: MouseButton, x: f32, y: f32) {}

    fn char_event(
        &mut self,
        _ctx: &mut Context,
        character: char,
        _keymods: KeyMods,
        _repeat: bool,
    ) {
    }

    fn key_down_event(
        &mut self,
        ctx: &mut Context,
        keycode: KeyCode,
        keymods: KeyMods,
        _repeat: bool,
    ) {
    }

    fn key_up_event(&mut self, _ctx: &mut Context, keycode: KeyCode, keymods: KeyMods) {}
}

fn main() {
    let conf = conf::Conf {
        high_dpi: true,
        ..Default::default()
    };
    start(conf, |mut ctx| UserData::owning(Stage::new(&mut ctx), ctx));
}

mod shader {
    use miniquad::{ShaderMeta, UniformBlockLayout, UniformDesc, UniformType};

    pub const VERTEX: &str = r#"
    #version 100
    uniform vec2 u_screen_size;

    attribute vec2 a_pos;
    attribute vec2 a_tc;
    attribute vec4 a_srgba;

    varying vec2 v_tc;
    varying vec4 v_rgba;

    // 0-1 linear  from  0-255 sRGB
    vec3 linear_from_srgb(vec3 srgb) {
        bvec3 cutoff = lessThan(srgb, vec3(10.31475));
        vec3 lower = srgb / vec3(3294.6);
        vec3 higher = pow((srgb + vec3(14.025)) / vec3(269.025), vec3(2.4));
        return mix(higher, lower, vec3(cutoff));
    }

    // 0-1 linear  from  0-255 sRGBA
    vec4 linear_from_srgba(vec4 srgba) {
        return vec4(linear_from_srgb(srgba.rgb), srgba.a / 255.0);
    }

    void main() {
        gl_Position = vec4(
            2.0 * a_pos.x / u_screen_size.x - 1.0,
            1.0 - 2.0 * a_pos.y / u_screen_size.y,
            0.0,
            1.0);

        v_tc = a_tc;
        v_rgba = linear_from_srgba(a_srgba);
    }
    "#;

    pub const FRAGMENT: &str = r#"
    #version 100
    uniform sampler2D u_sampler;
    precision highp float;

    varying vec2 v_tc;
    varying vec4 v_rgba;

    // 0-1 linear  from  0-255 sRGB
    vec3 linear_from_srgb(vec3 srgb) {
        bvec3 cutoff = lessThan(srgb, vec3(10.31475));
        vec3 lower = srgb / vec3(3294.6);
        vec3 higher = pow((srgb + vec3(14.025)) / vec3(269.025), vec3(2.4));
        return mix(higher, lower, vec3(cutoff));
    }

    // 0-1 linear  from  0-255 sRGBA
    vec4 linear_from_srgba(vec4 srgba) {
        return vec4(linear_from_srgb(srgba.rgb), srgba.a / 255.0);
    }

    // 0-255 sRGB  from  0-1 linear
    vec3 srgb_from_linear(vec3 rgb) {
        bvec3 cutoff = lessThan(rgb, vec3(0.0031308));
        vec3 lower = rgb * vec3(3294.6);
        vec3 higher = vec3(269.025) * pow(rgb, vec3(1.0 / 2.4)) - vec3(14.025);
        return mix(higher, lower, vec3(cutoff));
    }

    // 0-255 sRGBA  from  0-1 linear
    vec4 srgba_from_linear(vec4 rgba) {
        return vec4(srgb_from_linear(rgba.rgb), 255.0 * rgba.a);
    }

    void main() {
        vec4 texture_srgba = texture2D(u_sampler, v_tc);
        vec4 texture_rgba = linear_from_srgba(texture2D(u_sampler, v_tc) * 255.0); // TODO: sRGBA aware sampeler, see linear_from_srgb;
        gl_FragColor = v_rgba * texture_rgba;

        // miniquad doesn't support linear blending in the framebuffer.
        // so we need to convert linear to sRGBA:
        gl_FragColor = srgba_from_linear(gl_FragColor) / 255.0; // TODO: sRGBA aware framebuffer

        // We also apply this hack to at least get a bit closer to the desired blending:
        gl_FragColor.a = pow(gl_FragColor.a, 1.6); // Empiric nonsense
    }
    "#;

    pub fn meta() -> ShaderMeta {
        ShaderMeta {
            images: vec!["u_sampler".to_string()],
            uniforms: UniformBlockLayout {
                uniforms: vec![UniformDesc::new("u_screen_size", UniformType::Float2)],
            },
        }
    }

    #[repr(C)]
    #[derive(Debug)]
    pub struct Uniforms {
        pub u_screen_size: (f32, f32),
    }
}
