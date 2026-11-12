//! Viewer side: receives frames, shows them with Vulkan, and forwards mouse and keyboard input.

use crate::render::{fit, Renderer};
use crate::{event, keys, mac, prompt, Result, EV_BUTTON, EV_KEY, EV_MOVE, EV_SCROLL, MAGIC};
use hmac::Mac;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy};
use winit::keyboard::{Key, KeyCode, ModifiersState, NamedKey, PhysicalKey};
use winit::platform::modifier_supplement::KeyEventExtModifierSupplement;
use winit::window::{Fullscreen, Window, WindowId};

/// The latest remote screen contents, tightly packed BGRA.
#[derive(Default)]
struct Frame {
    w: u32,
    h: u32,
    data: Vec<u8>,
    dirty: Option<(u32, u32)>, // rows [start, end) that changed since the last upload to the GPU
}

enum Wake {
    Frame,
    Closed(String),
}

/// Connects and authenticates. The password is only needed (and only asked for) if
/// the host has one set.
fn connect(addr: SocketAddr, mut password: Option<String>) -> Result<TcpStream> {
    let mut asked = false;
    loop {
        let mut stream = TcpStream::connect_timeout(&addr, Duration::from_secs(3))?;
        stream.set_nodelay(true)?;
        let mut hello = [0u8; 23];
        stream.read_exact(&mut hello)?;
        if &hello[..6] != MAGIC {
            return Err("that is not a wrdesk host (or it runs a different version)".into());
        }
        let locked = hello[22] != 0;
        if locked && password.is_none() {
            // The host does not wait long for the answer, so ask first and reconnect.
            password = Some(prompt("Password: ")?);
            asked = true;
            continue;
        }
        let key = if locked { password.as_deref().unwrap_or("") } else { "" };
        stream.write_all(&mac(key, &hello[6..22]).finalize().into_bytes())?;
        let mut ok = [0u8; 1];
        stream.read_exact(&mut ok)?;
        if ok[0] == 1 {
            return Ok(stream);
        }
        if !asked {
            return Err("wrong password".into());
        }
        println!("Wrong password.");
        password = None;
    }
}

pub fn run(addr: SocketAddr, name: &str, password: Option<String>, view_only: bool) -> Result<()> {
    let stream = connect(addr, password)?;

    let frame = Arc::new(Mutex::new(Frame::default()));
    let event_loop = EventLoop::<Wake>::with_user_event().build()?;

    let proxy = event_loop.create_proxy();
    let (reader, shared) = (stream.try_clone()?, frame.clone());
    thread::spawn(move || {
        let reason = match receive(reader, &shared, &proxy) {
            Ok(()) => return, // the window closed first
            Err(e) => e.to_string(),
        };
        let _ = proxy.send_event(Wake::Closed(reason));
    });

    let mut app = App {
        renderer: None,
        window: None,
        frame,
        stream,
        view_only,
        title: format!("wrdesk - {name}"),
        remote: (0, 0),
        modifiers: ModifiersState::empty(),
        scroll: (0.0, 0.0),
        error: None,
    };
    event_loop.run_app(&mut app)?;
    app.error.map_or(Ok(()), Err)
}

/// Network thread: applies each incoming delta to the shared frame and wakes the window.
fn receive(mut stream: TcpStream, shared: &Mutex<Frame>, proxy: &EventLoopProxy<Wake>) -> Result<()> {
    let mut header = [0u8; 20];
    let (mut packed, mut delta) = (Vec::new(), Vec::new());
    loop {
        stream.read_exact(&mut header)?;
        let field = |i: usize| u32::from_le_bytes([header[i], header[i + 1], header[i + 2], header[i + 3]]);
        let (w, h, y, rows, n) = (field(0), field(4), field(8), field(12), field(16) as usize);
        if w == 0 || h == 0 || w > 16384 || h > 16384 || rows == 0 || y >= h || rows > h - y {
            return Err("bad frame size".into());
        }
        let row = w as usize * 4;
        let len = rows as usize * row;
        if n > lz4_flex::block::get_maximum_output_size(len) {
            return Err("bad frame length".into());
        }
        packed.resize(n, 0);
        stream.read_exact(&mut packed)?;
        delta.resize(len, 0);
        if lz4_flex::block::decompress_into(&packed, &mut delta)? != len {
            return Err("truncated frame".into());
        }
        {
            let mut frame = shared.lock().unwrap();
            if (frame.w, frame.h) != (w, h) {
                *frame = Frame { w, h, data: vec![0; h as usize * row], dirty: None };
            }
            for (pixel, d) in frame.data[y as usize * row..].iter_mut().zip(&delta) {
                *pixel ^= d;
            }
            let (start, end) = frame.dirty.unwrap_or((y, y + rows));
            frame.dirty = Some((start.min(y), end.max(y + rows)));
        }
        if proxy.send_event(Wake::Frame).is_err() {
            return Ok(());
        }
    }
}

struct App {
    renderer: Option<Renderer>, // declared before `window` so it is dropped first
    window: Option<Arc<Window>>,
    frame: Arc<Mutex<Frame>>,
    stream: TcpStream,
    view_only: bool,
    title: String,
    remote: (u32, u32),
    modifiers: ModifiersState,
    scroll: (f64, f64),
    error: Option<Box<dyn std::error::Error + Send + Sync>>,
}

impl App {
    fn send(&mut self, kind: u8, a: u8, b: u8, payload: u32) {
        if !self.view_only {
            let _ = self.stream.write_all(&event(kind, a, b, payload)); // the reader thread reports a dead link
        }
    }

    fn fail(&mut self, event_loop: &ActiveEventLoop, error: Box<dyn std::error::Error + Send + Sync>) {
        self.error = Some(error);
        event_loop.exit();
    }

    /// Sends whole scroll lines, carrying the fractional remainder over to the next event.
    fn scroll(&mut self, dx: f64, dy: f64) {
        self.scroll.0 += dx;
        self.scroll.1 += dy;
        let (x, y) = (self.scroll.0.trunc(), self.scroll.1.trunc());
        self.scroll.0 -= x;
        self.scroll.1 -= y;
        // winit: positive = content moves right/down. Host: positive = scroll right/down.
        if y != 0.0 {
            self.send(EV_SCROLL, 0, 0, (-y as i32) as u32);
        }
        if x != 0.0 {
            self.send(EV_SCROLL, 1, 0, (-x as i32) as u32);
        }
    }
}

impl ApplicationHandler<Wake> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attributes = Window::default_attributes()
            .with_title(&self.title)
            .with_inner_size(LogicalSize::new(1280.0, 720.0));
        let window = match event_loop.create_window(attributes) {
            Ok(window) => Arc::new(window),
            Err(e) => return self.fail(event_loop, e.into()),
        };
        match Renderer::new(&window) {
            Ok(renderer) => self.renderer = Some(renderer),
            Err(e) => return self.fail(event_loop, e),
        }
        self.window = Some(window);
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, wake: Wake) {
        match wake {
            Wake::Frame => {
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            Wake::Closed(reason) => self.fail(event_loop, format!("connection closed: {reason}").into()),
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let (Some(window), Some(renderer)) = (self.window.clone(), self.renderer.as_mut()) else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Err(e) = renderer.resize(size.width, size.height) {
                    return self.fail(event_loop, e);
                }
                window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                let mut frame = self.frame.lock().unwrap();
                if frame.w == 0 {
                    return;
                }
                self.remote = (frame.w, frame.h);
                match renderer.render(&frame.data, frame.w, frame.h, frame.dirty) {
                    Ok(true) => frame.dirty = None,
                    Ok(false) => {} // not drawn (minimized); keep the dirty rows for later
                    Err(e) => {
                        drop(frame);
                        self.fail(event_loop, e);
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.remote.0 == 0 {
                    return;
                }
                // Map the window position onto the letterboxed remote image.
                let size = window.inner_size();
                let (x, y, w, h) = fit((size.width, size.height), self.remote);
                let nx = ((position.x - x as f64) / w as f64).clamp(0.0, 1.0);
                let ny = ((position.y - y as f64) / h as f64).clamp(0.0, 1.0);
                let payload = ((nx * 65535.0) as u32) << 16 | (ny * 65535.0) as u32;
                self.send(EV_MOVE, 0, 0, payload);
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let button = match button {
                    MouseButton::Left => 0,
                    MouseButton::Right => 1,
                    MouseButton::Middle => 2,
                    _ => return,
                };
                self.send(EV_BUTTON, button, (state == ElementState::Pressed) as u8, 0);
            }
            WindowEvent::MouseWheel { delta, .. } => match delta {
                MouseScrollDelta::LineDelta(x, y) => self.scroll(x as f64, y as f64),
                MouseScrollDelta::PixelDelta(p) => self.scroll(p.x / 40.0, p.y / 40.0),
            },
            WindowEvent::ModifiersChanged(modifiers) => self.modifiers = modifiers.state(),
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                // Ctrl+Alt+F toggles fullscreen locally instead of going to the host.
                if event.physical_key == PhysicalKey::Code(KeyCode::KeyF)
                    && self.modifiers.control_key()
                    && self.modifiers.alt_key()
                {
                    if pressed && !event.repeat {
                        let fullscreen = window.fullscreen().is_none().then_some(Fullscreen::Borderless(None));
                        window.set_fullscreen(fullscreen);
                    }
                    return;
                }
                // The key as it would be typed without modifiers held, so the host's own
                // modifier state (which we mirror) produces the final character.
                let (kind, code) = match event.key_without_modifiers() {
                    Key::Character(text) => match text.chars().next() {
                        Some(c) => (0, c as u32),
                        None => return,
                    },
                    Key::Named(named) => match named_key(named) {
                        Some(id) => (1, id),
                        None => return,
                    },
                    _ => return,
                };
                self.send(EV_KEY, kind, pressed as u8, code);
            }
            WindowEvent::Focused(false) => {
                // Don't leave modifiers stuck down on the host (e.g. after Alt+Tab away).
                for id in [keys::SHIFT, keys::CONTROL, keys::ALT, keys::META] {
                    self.send(EV_KEY, 1, 0, id);
                }
            }
            _ => {}
        }
    }
}

fn named_key(key: NamedKey) -> Option<u32> {
    Some(match key {
        NamedKey::Enter => keys::ENTER,
        NamedKey::Tab => keys::TAB,
        NamedKey::Space => keys::SPACE,
        NamedKey::Backspace => keys::BACKSPACE,
        NamedKey::Escape => keys::ESCAPE,
        NamedKey::Delete => keys::DELETE,
        NamedKey::Insert => keys::INSERT,
        NamedKey::Home => keys::HOME,
        NamedKey::End => keys::END,
        NamedKey::PageUp => keys::PAGE_UP,
        NamedKey::PageDown => keys::PAGE_DOWN,
        NamedKey::ArrowLeft => keys::LEFT,
        NamedKey::ArrowRight => keys::RIGHT,
        NamedKey::ArrowUp => keys::UP,
        NamedKey::ArrowDown => keys::DOWN,
        NamedKey::Shift => keys::SHIFT,
        NamedKey::Control => keys::CONTROL,
        NamedKey::Alt => keys::ALT,
        NamedKey::Super => keys::META,
        NamedKey::CapsLock => keys::CAPS_LOCK,
        NamedKey::F1 => keys::F1,
        NamedKey::F2 => keys::F1 + 1,
        NamedKey::F3 => keys::F1 + 2,
        NamedKey::F4 => keys::F1 + 3,
        NamedKey::F5 => keys::F1 + 4,
        NamedKey::F6 => keys::F1 + 5,
        NamedKey::F7 => keys::F1 + 6,
        NamedKey::F8 => keys::F1 + 7,
        NamedKey::F9 => keys::F1 + 8,
        NamedKey::F10 => keys::F1 + 9,
        NamedKey::F11 => keys::F1 + 10,
        NamedKey::F12 => keys::F1 + 11,
        _ => return None,
    })
}
