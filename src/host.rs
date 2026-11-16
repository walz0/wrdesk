//! Host side: captures the primary display, streams it, and injects the client's input.

use crate::{
    audio, keys, mac, web, Result, ANNOUNCE, AUDIO, DISCOVER, EV_BUTTON, EV_GAME, EV_KEY, EV_MOTION, EV_MOVE, EV_SCROLL,
    GAME_FPS, MAGIC, PORT, SOUND,
};
use enigo::{Axis, Button, Coordinate, Direction, Enigo, Key, Keyboard, Mouse, Settings};
use hmac::Mac;
use scrap::{Capturer, Display};
use std::io::{ErrorKind, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream, UdpSocket};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

const HEADER: usize = 20;
/// An idle host sends an empty header this often, so clients can tell a still screen from a dead link.
const KEEPALIVE: Duration = Duration::from_secs(1);
/// How often the screen is checked for a new frame once one is due: in game mode, and otherwise.
const POLL_GAME: Duration = Duration::from_millis(1);
const POLL: Duration = Duration::from_millis(8);

#[derive(Clone, Copy)]
pub struct Options {
    pub fps: u32,
    pub scale: u32,
    pub mbps: f64, // 0 = unlimited
    pub web: bool, // also serve the browser client on WEB_PORT
}

/// A connection to a client: a plain socket, or a browser's HTTP requests.
pub trait Link: Read + Write + Send + Sized + 'static {
    fn socket(&self) -> &TcpStream;
    fn try_clone(&self) -> std::io::Result<Self>;
}

impl Link for TcpStream {
    fn socket(&self) -> &TcpStream {
        self
    }
    fn try_clone(&self) -> std::io::Result<Self> {
        TcpStream::try_clone(self)
    }
}

pub fn run(password: &str, options: Options) -> Result<()> {
    let listener = TcpListener::bind(("0.0.0.0", PORT))?;
    let name = gethostname::gethostname().to_string_lossy().into_owned();

    // Answer discovery broadcasts so clients can find us without an address.
    let udp = UdpSocket::bind(("0.0.0.0", PORT))?;
    let locked = !password.is_empty();
    let reply = [ANNOUNCE, &[locked as u8], name.as_bytes()].concat();
    thread::spawn(move || {
        let mut buf = [0u8; 64];
        loop {
            if let Ok((n, from)) = udp.recv_from(&mut buf) {
                if &buf[..n] == DISCOVER {
                    let _ = udp.send_to(&reply, from);
                }
            }
        }
    });

    println!("wrdesk {} hosting '{name}' on port {PORT}. Press Ctrl+C to stop.", env!("CARGO_PKG_VERSION"));
    if options.web {
        if let Err(e) = web::listen(password, options) {
            eprintln!("no web client: {e}");
        }
    }
    if !locked {
        println!("No password set: anyone on this network can view and control this computer (-p sets one).");
    }
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let password = password.to_string();
        thread::spawn(move || session(stream, &password, options));
    }
    Ok(())
}

/// Serves one connection of a client until it leaves, logging the client's coming and going.
pub fn session(mut stream: impl Link, password: &str, options: Options) {
    let peer = stream.socket().peer_addr().map(|a| a.ip().to_string()).unwrap_or_default();
    let result = match authenticate(&mut stream, password) {
        // A client's audio comes and goes with its screen connection, which does the logging.
        Ok(SOUND) => {
            if let Err(e) = send_audio(stream) {
                eprintln!("no audio for {peer}: {e}");
            }
            return;
        }
        Ok(_) => serve(stream, &peer, options),
        Err(e) => Err(e),
    };
    match result {
        Ok(()) => println!("{peer} disconnected"),
        Err(e) => println!("{peer} disconnected: {e}"),
    }
}

/// Checks the client's password, returning what it wants the connection for.
fn authenticate(stream: &mut impl Link, password: &str) -> Result<u8> {
    stream.socket().set_nodelay(true)?;
    stream.socket().set_read_timeout(Some(Duration::from_secs(10)))?;

    let mut nonce = [0u8; 16];
    getrandom::getrandom(&mut nonce).map_err(|e| e.to_string())?;
    stream.write_all(&[&MAGIC[..], &nonce, &[!password.is_empty() as u8]].concat())?;
    let mut answer = [0u8; 33]; // the HMAC, then the connection kind
    stream.read_exact(&mut answer)?;
    if mac(password, &nonce).verify_slice(&answer[..32]).is_err() {
        thread::sleep(Duration::from_secs(1)); // slows down password guessing
        stream.write_all(&[0])?;
        return Err("wrong password".into());
    }
    stream.write_all(&[1])?;
    stream.socket().set_read_timeout(None)?;
    // A client that vanished (e.g. a laptop that went to sleep) stops acknowledging data;
    // give up on it instead of holding the capture for minutes.
    stream.socket().set_write_timeout(Some(Duration::from_secs(10)))?;
    Ok(answer[32])
}

/// Streams what the speakers play until the client goes away.
fn send_audio(mut stream: impl Write) -> Result<()> {
    let (mut packet, mut last_sent, mut gone) = (Vec::new(), Instant::now(), false);
    let result = audio::capture(|rate, channels, samples| {
        if samples.is_empty() && last_sent.elapsed() < KEEPALIVE {
            return Ok(());
        }
        packet.clear();
        // Silence is sent as a keepalive, so the client can tell it from a dead link.
        let header = if samples.is_empty() { [0; 5] } else { [0, AUDIO, rate, channels as u32, samples.len() as u32 * 4] };
        for value in header {
            packet.extend_from_slice(&value.to_le_bytes());
        }
        packet.extend(samples.iter().flat_map(|s| s.to_le_bytes()));
        stream.write_all(&packet).inspect_err(|_| gone = true)?;
        last_sent = Instant::now();
        Ok(())
    });
    if gone { Ok(()) } else { result }
}

/// Streams the screen and injects the client's input until it leaves.
fn serve(stream: impl Link, peer: &str, options: Options) -> Result<()> {
    println!("{peer} connected");

    let alive = Arc::new(AtomicBool::new(true));
    let game = Arc::new(AtomicBool::new(false)); // the client has the mouse captured, for a game
    let input = stream.try_clone()?;
    let (input_alive, input_game) = (alive.clone(), game.clone());
    thread::spawn(move || {
        if let Err(e) = input_loop(input, &input_game) {
            eprintln!("cannot inject input: {e}");
        }
        input_alive.store(false, Ordering::Relaxed);
    });

    let result = capture_loop(stream.try_clone()?, peer, &alive, &game, options);
    let _ = stream.socket().shutdown(Shutdown::Both); // also ends the input thread
    result
}

fn capture_loop(mut stream: impl Write, peer: &str, alive: &AtomicBool, game: &AtomicBool, options: Options) -> Result<()> {
    let scale = options.scale as usize;
    let mut capturer: Option<Capturer> = None;
    // `prev` is the last frame the client has; `scaled` is only used when downscaling.
    let (mut prev, mut scaled, mut delta, mut out) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut size = (0, 0);
    let mut force = false;
    let mut last_sent = Instant::now();
    // When the next frame may be captured, which holds the frame rate (and --mbps).
    let mut due = Instant::now();
    let mut gaming = false;

    while alive.load(Ordering::Relaxed) {
        if game.load(Ordering::Relaxed) != gaming {
            gaming = !gaming;
            match gaming {
                true => println!("{peer} captured the mouse: game mode, {} fps", options.fps.max(GAME_FPS)),
                false => println!("{peer} released the mouse"),
            }
        }
        let fps = if gaming { options.fps.max(GAME_FPS) } else { options.fps };
        let frame_time = Duration::from_secs_f64(1.0 / fps as f64);

        let start = Instant::now();
        if last_sent.elapsed() >= KEEPALIVE {
            stream.write_all(&[0; HEADER])?;
            last_sent = start;
        }
        if start < due {
            thread::sleep((due - start).min(KEEPALIVE));
            continue;
        }
        let cap = match &mut capturer {
            Some(cap) => cap,
            None => match Display::primary().and_then(Capturer::new) {
                Ok(cap) => capturer.insert(cap),
                Err(_) => {
                    thread::sleep(Duration::from_millis(250));
                    continue;
                }
            },
        };

        let (full_w, full_h) = (cap.width(), cap.height());
        let (w, h) = ((full_w / scale).max(1), (full_h / scale).max(1));
        let row = w * 4;
        // Ok(bytes sent) or the capture error.
        let sent = match cap.frame() {
            Ok(frame) => {
                if (w, h) != size {
                    size = (w, h);
                    prev = vec![0u8; row * h];
                    delta = prev.clone();
                    scaled = if scale > 1 { prev.clone() } else { Vec::new() };
                    out = vec![0u8; HEADER + lz4_flex::block::get_maximum_output_size(prev.len())];
                    force = true;
                }
                // `pixels` is BGRA with `stride` bytes per row: the capture itself, or a
                // cheap nearest-neighbour downscale of it.
                let full_stride = frame.len() / full_h;
                let (pixels, stride): (&[u8], usize) = if scale == 1 {
                    (&frame, full_stride)
                } else {
                    for (y, dst) in scaled.chunks_exact_mut(row).enumerate() {
                        let src = &frame[y * scale * full_stride..][..w * scale * 4];
                        for (d, s) in dst.chunks_exact_mut(4).zip(src.chunks_exact(scale * 4)) {
                            d.copy_from_slice(&s[..4]);
                        }
                    }
                    (&scaled, row)
                };

                // Find the band of rows that changed; everything else costs one memcmp.
                let differs = |y: usize| pixels[y * stride..][..row] != prev[y * row..][..row];
                let band = if force {
                    Some((0, h))
                } else {
                    (0..h).find(|&y| differs(y)).map(|y0| (y0, (y0..h).rfind(|&y| differs(y)).unwrap_or(y0) + 1))
                };
                match band {
                    None => Ok(0),
                    Some((y0, y1)) => {
                        // XOR against the previous frame is mostly zeros, which lz4 crushes.
                        for y in y0..y1 {
                            let src = &pixels[y * stride..][..row];
                            let old = &mut prev[y * row..][..row];
                            let d = &mut delta[(y - y0) * row..][..row];
                            for ((d, o), s) in d.iter_mut().zip(old.iter_mut()).zip(src) {
                                *d = *s ^ *o;
                                *o = *s;
                            }
                        }
                        let n = lz4_flex::block::compress_into(&delta[..(y1 - y0) * row], &mut out[HEADER..])?;
                        for (i, value) in [w, h, y0, y1 - y0, n].into_iter().enumerate() {
                            out[i * 4..i * 4 + 4].copy_from_slice(&(value as u32).to_le_bytes());
                        }
                        stream.write_all(&out[..HEADER + n])?;
                        last_sent = Instant::now();
                        force = false;
                        Ok(HEADER + n)
                    }
                }
            }
            Err(e) => Err(e.kind()),
        };

        match sent {
            // Hold the frame rate, and stretch the gap after big frames to respect --mbps.
            Ok(bytes) => {
                let mut interval = frame_time;
                if options.mbps > 0.0 {
                    interval = interval.max(Duration::from_secs_f64(bytes as f64 * 8.0 / (options.mbps * 1e6)));
                }
                // The schedule may slip a quarter frame behind, so the next frame can be taken
                // as soon as it appears rather than up to a frame late; the rate still holds.
                due = due.max(start.checked_sub(frame_time / 4).unwrap_or(start)) + interval;
            }
            // Nothing new on screen yet: look again soon. A game wants the frame the moment it is drawn.
            Err(ErrorKind::WouldBlock) => thread::sleep(if gaming { POLL_GAME } else { POLL }),
            // Resolution change, secure desktop, etc: start capturing again.
            Err(_) => {
                capturer = None;
                thread::sleep(Duration::from_millis(250));
            }
        }
    }
    Ok(())
}

fn input_loop(mut stream: impl Read, game: &AtomicBool) -> Result<()> {
    // Relative motion must reach Windows as relative motion, for the games that read it raw:
    // enigo otherwise turns it into a move to an absolute position.
    let settings = Settings { windows_subject_to_mouse_speed_and_acceleration_level: true, ..Settings::default() };
    let mut enigo = Enigo::new(&settings).map_err(|e| e.to_string())?;
    let direction = |pressed: u8| if pressed != 0 { Direction::Press } else { Direction::Release };
    let mut modifiers = [false; 4]; // Shift, Control, Alt and Meta, as the client holds them
    #[cfg(windows)]
    let mut down = Vec::new();
    let mut ev = [0u8; 8];
    loop {
        if stream.read_exact(&mut ev).is_err() {
            return Ok(()); // client went away
        }
        let payload = u32::from_le_bytes([ev[4], ev[5], ev[6], ev[7]]);
        let result = match ev[0] {
            EV_MOVE => enigo.main_display().and_then(|(w, h)| {
                let x = (payload >> 16) as i64 * (w as i64 - 1) / 65535;
                let y = (payload & 0xffff) as i64 * (h as i64 - 1) / 65535;
                enigo.move_mouse(x as i32, y as i32, Coordinate::Abs)
            }),
            EV_MOTION => enigo.move_mouse((payload >> 16) as i16 as i32, payload as i16 as i32, Coordinate::Rel),
            EV_BUTTON => {
                let button = match ev[1] {
                    0 => Button::Left,
                    1 => Button::Right,
                    _ => Button::Middle,
                };
                enigo.button(button, direction(ev[2]))
            }
            EV_GAME => {
                game.store(ev[1] != 0, Ordering::Relaxed);
                Ok(())
            }
            EV_SCROLL => enigo.scroll(payload as i32, if ev[1] == 0 { Axis::Vertical } else { Axis::Horizontal }),
            EV_KEY => {
                if ev[1] == 1 && (keys::SHIFT..=keys::META).contains(&payload) {
                    modifiers[(payload - keys::SHIFT) as usize] = ev[2] != 0;
                }
                match (ev[1], char::from_u32(payload)) {
                    #[cfg(windows)]
                    (2, Some(c)) => produced(&mut enigo, c, ev[2] != 0, modifiers, &mut down),
                    _ => match key(ev[1], payload) {
                        Some(key) => enigo.key(key, direction(ev[2])),
                        None => Ok(()),
                    },
                }
            }
            _ => Ok(()),
        };
        if let Err(e) = result {
            eprintln!("input: {e}");
        }
    }
}

/// Enters a character that has its modifiers applied already ('A', not 'a' with Shift), with
/// `down` the keys this has pressed. enigo cannot: it presses no key at all for a shifted one.
///
/// The character's key is pressed if the Shift the client holds is the one the character takes,
/// or if other modifiers make it a shortcut. Otherwise the character is entered as text.
#[cfg(windows)]
fn produced(enigo: &mut Enigo, c: char, pressed: bool, modifiers: [bool; 4], down: &mut Vec<u32>) -> enigo::InputResult<()> {
    #[link(name = "user32")]
    unsafe extern "system" {
        fn VkKeyScanW(ch: u16) -> i16; // key | modifiers << 8 (1 = Shift, 2 = Control, 4 = Alt), or -1
    }
    let found = match *c.encode_utf16(&mut [0; 2]) {
        [unit] => unsafe { VkKeyScanW(unit) },
        _ => -1,
    };
    let key = found as u32 & 0xff;
    if !pressed {
        // The key may be released under another Shift, so as another character, than it was pressed.
        return match down.iter().position(|&held| held == key) {
            Some(at) if found != -1 => {
                down.swap_remove(at);
                enigo.key(Key::Other(key), Direction::Release)
            }
            _ => Ok(()),
        };
    }
    let [shift, control, alt, meta] = modifiers;
    if found != -1 && (control || alt || meta || found >> 8 == shift as i16) {
        if !down.contains(&key) {
            down.push(key);
        }
        enigo.key(Key::Other(key), Direction::Press)
    } else {
        enigo.text(c.encode_utf8(&mut [0; 4]))
    }
}

fn key(kind: u8, code: u32) -> Option<Key> {
    if kind != 1 {
        return char::from_u32(code).map(Key::Unicode);
    }
    Some(match code {
        keys::ENTER => Key::Return,
        keys::TAB => Key::Tab,
        keys::SPACE => Key::Space,
        keys::BACKSPACE => Key::Backspace,
        keys::ESCAPE => Key::Escape,
        keys::DELETE => Key::Delete,
        #[cfg(not(target_os = "macos"))]
        keys::INSERT => Key::Insert,
        keys::HOME => Key::Home,
        keys::END => Key::End,
        keys::PAGE_UP => Key::PageUp,
        keys::PAGE_DOWN => Key::PageDown,
        keys::LEFT => Key::LeftArrow,
        keys::RIGHT => Key::RightArrow,
        keys::UP => Key::UpArrow,
        keys::DOWN => Key::DownArrow,
        keys::SHIFT => Key::Shift,
        keys::CONTROL => Key::Control,
        keys::ALT => Key::Alt,
        keys::META => Key::Meta,
        keys::CAPS_LOCK => Key::CapsLock,
        #[cfg(windows)]
        keys::PRINT_SCREEN => Key::PrintScr,
        #[cfg(windows)]
        keys::MENU => Key::Apps,
        #[cfg(windows)]
        keys::PAUSE => Key::Pause,
        #[cfg(windows)]
        keys::SCROLL_LOCK => Key::Scroll,
        #[cfg(windows)]
        keys::NUM_LOCK => Key::Numlock,
        101 => Key::F1,
        102 => Key::F2,
        103 => Key::F3,
        104 => Key::F4,
        105 => Key::F5,
        106 => Key::F6,
        107 => Key::F7,
        108 => Key::F8,
        109 => Key::F9,
        110 => Key::F10,
        111 => Key::F11,
        112 => Key::F12,
        _ => return None,
    })
}
