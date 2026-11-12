//! wrdesk - a tiny LAN remote desktop.
//!
//!   wrdesk host [-p PASSWORD] [--fps N] [--scale N] [--mbps N]   share this machine
//!   wrdesk join [TARGET] [-p PASSWORD]     view and control a host (name or IP; omit to auto-discover)
//!   wrdesk                                 ask whether to host or join
//!
//! Wire protocol (TCP, little endian):
//!   host -> client   MAGIC, 16 byte nonce
//!   client -> host   HMAC-SHA256(password, nonce)
//!   host -> client   1 byte (1 = ok), then frames: [u32 w][u32 h][u32 y][u32 rows][u32 len]
//!                    followed by lz4(rows y..y+rows of the frame XOR the previous frame)
//!   client -> host   8 byte input events, see `event`

mod client;
mod host;
mod render;

use hmac::{Hmac, Mac};
use std::io::Write;
use std::net::{IpAddr, SocketAddr, ToSocketAddrs, UdpSocket};
use std::time::{Duration, Instant};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

pub const PORT: u16 = 47110;
pub const MAGIC: &[u8; 6] = b"WRDSK1";
pub const DISCOVER: &[u8] = b"WRDESK?";
pub const ANNOUNCE: &[u8] = b"WRDESK!";

pub const EV_MOVE: u8 = 1; // payload = x << 16 | y, each 0..=65535 across the remote screen
pub const EV_BUTTON: u8 = 2; // a = 0 left / 1 right / 2 middle, b = pressed
pub const EV_SCROLL: u8 = 3; // a = 0 vertical / 1 horizontal, payload = lines as i32
pub const EV_KEY: u8 = 4; // a = 0 unicode char / 1 named key (see `keys`), b = pressed

pub fn event(kind: u8, a: u8, b: u8, payload: u32) -> [u8; 8] {
    let p = payload.to_le_bytes();
    [kind, a, b, 0, p[0], p[1], p[2], p[3]]
}

/// Ids for non-character keys.
pub mod keys {
    pub const ENTER: u32 = 1;
    pub const TAB: u32 = 2;
    pub const SPACE: u32 = 3;
    pub const BACKSPACE: u32 = 4;
    pub const ESCAPE: u32 = 5;
    pub const DELETE: u32 = 6;
    pub const INSERT: u32 = 7;
    pub const HOME: u32 = 8;
    pub const END: u32 = 9;
    pub const PAGE_UP: u32 = 10;
    pub const PAGE_DOWN: u32 = 11;
    pub const LEFT: u32 = 12;
    pub const RIGHT: u32 = 13;
    pub const UP: u32 = 14;
    pub const DOWN: u32 = 15;
    pub const SHIFT: u32 = 16;
    pub const CONTROL: u32 = 17;
    pub const ALT: u32 = 18;
    pub const META: u32 = 19;
    pub const CAPS_LOCK: u32 = 20;
    pub const F1: u32 = 101; // F1..=F12 are 101..=112
}

pub fn mac(password: &str, nonce: &[u8]) -> Hmac<sha2::Sha256> {
    let mut mac = Hmac::<sha2::Sha256>::new_from_slice(password.as_bytes()).expect("any key length works");
    mac.update(nonce);
    mac
}

/// Broadcasts on the LAN and collects every host that answers.
fn discover() -> Result<Vec<(String, IpAddr)>> {
    let socket = UdpSocket::bind(("0.0.0.0", 0))?;
    socket.set_broadcast(true)?;
    socket.set_read_timeout(Some(Duration::from_millis(50)))?;
    socket.send_to(DISCOVER, ("255.255.255.255", PORT))?;
    let deadline = Instant::now() + Duration::from_millis(400);
    let mut hosts: Vec<(String, IpAddr)> = Vec::new();
    let mut buf = [0u8; 256];
    while Instant::now() < deadline {
        let Ok((n, from)) = socket.recv_from(&mut buf) else { continue };
        if let Some(name) = buf[..n].strip_prefix(ANNOUNCE) {
            if !hosts.iter().any(|(_, ip)| *ip == from.ip()) {
                hosts.push((String::from_utf8_lossy(name).into_owned(), from.ip()));
            }
        }
    }
    Ok(hosts)
}

fn prompt(text: &str) -> Result<String> {
    print!("{text}");
    std::io::stdout().flush()?;
    let mut line = String::new();
    std::io::stdin().read_line(&mut line)?;
    Ok(line.trim().to_string())
}

/// Turns the optional command line target into a (name, address) to connect to.
fn resolve(target: Option<&str>) -> Result<(String, SocketAddr)> {
    if let Some(ip) = target.and_then(|t| t.parse::<IpAddr>().ok()) {
        return Ok((ip.to_string(), SocketAddr::new(ip, PORT)));
    }
    let hosts = discover()?;
    if let Some(target) = target {
        if let Some((name, ip)) = hosts.iter().find(|(name, _)| name.eq_ignore_ascii_case(target)) {
            return Ok((name.clone(), SocketAddr::new(*ip, PORT)));
        }
        let addr = (target, PORT).to_socket_addrs().ok().and_then(|mut a| a.next());
        return Ok((target.to_string(), addr.ok_or_else(|| format!("no host named '{target}' found"))?));
    }
    let (name, ip) = match hosts.len() {
        0 => return Err("no hosts found on the network (try `wrdesk join <ip address>`)".into()),
        1 => hosts[0].clone(),
        _ => {
            for (i, (name, ip)) in hosts.iter().enumerate() {
                println!("  {}) {name}  ({ip})", i + 1);
            }
            let choice: usize = prompt("Connect to: ")?.parse().map_err(|_| "not a number")?;
            hosts.get(choice.wrapping_sub(1)).ok_or("no such host")?.clone()
        }
    };
    Ok((name, SocketAddr::new(ip, PORT)))
}

fn random_password() -> Result<String> {
    const CHARS: &[u8] = b"abcdefghjkmnpqrstuvwxyz23456789";
    let mut bytes = [0u8; 8];
    getrandom::getrandom(&mut bytes).map_err(|e| e.to_string())?;
    Ok(bytes.iter().map(|b| CHARS[*b as usize % CHARS.len()] as char).collect())
}

fn number(value: Option<String>, flag: &str, min: f64, max: f64) -> Result<f64> {
    let n: f64 = value.and_then(|v| v.parse().ok()).ok_or_else(|| format!("{flag} needs a number"))?;
    if n < min || n > max {
        return Err(format!("{flag} must be between {min} and {max}").into());
    }
    Ok(n)
}

fn run() -> Result<()> {
    let mut password = std::env::var("WRDESK_PASSWORD").ok().filter(|p| !p.is_empty());
    let mut view_only = false;
    let mut options = host::Options { fps: 30, scale: 1, mbps: 0.0 };
    let mut positional = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-p" | "--password" => password = Some(args.next().ok_or("-p needs a value")?),
            "--view-only" => view_only = true,
            "--fps" => options.fps = number(args.next(), "--fps", 1.0, 240.0)? as u32,
            "--scale" => options.scale = number(args.next(), "--scale", 1.0, 8.0)? as u32,
            "--mbps" => options.mbps = number(args.next(), "--mbps", 0.1, 100000.0)?,
            "-h" | "--help" => {
                println!(
                    "wrdesk host [-p PASSWORD] [options]          share this machine\n\
                     \x20   --fps N     frame rate limit (default 30)\n\
                     \x20   --scale N   send 1/N of the resolution (default 1; 2 = a quarter of the pixels)\n\
                     \x20   --mbps N    bandwidth limit in megabits per second (default unlimited)\n\
                     wrdesk join [TARGET] [-p PASSWORD] [--view-only]\n\
                     \x20                                            view and control a host (name or IP; omit to auto-discover)\n\
                     wrdesk                                       ask whether to host or join\n\n\
                     The password can also be set with the WRDESK_PASSWORD environment variable.\n\
                     In the viewer, Ctrl+Alt+F toggles fullscreen."
                );
                return Ok(());
            }
            _ => positional.push(arg),
        }
    }

    // Started without saying what to do (e.g. double-clicked): ask.
    let mut command = positional.first().cloned();
    let interactive = command.is_none();
    if interactive {
        println!("wrdesk\n  1) Host this machine\n  2) Join a session");
        command = Some(match prompt("Choose [1/2]: ")?.to_lowercase().as_str() {
            "1" | "h" | "host" => "host".to_string(),
            "2" | "j" | "join" => "join".to_string(),
            _ => return Err("expected 1 or 2".into()),
        });
    }
    let result = match command.as_deref() {
        // Hosting is headless: it never opens a window or shows the video feed.
        Some("host") => {
            if password.is_none() && interactive {
                password = Some(prompt("Set a password (Enter to generate one): ")?).filter(|p| !p.is_empty());
            }
            let password = match password {
                Some(p) => p,
                None => {
                    let p = random_password()?;
                    println!("One-time password: {p}");
                    p
                }
            };
            host::run(&password, options)
        }
        target => {
            // `wrdesk join [TARGET]`, or just `wrdesk TARGET`.
            let target = if target == Some("join") { positional.get(1).map(String::as_str) } else { target };
            join(target, password, view_only, interactive)
        }
    };
    if interactive {
        if let Err(e) = &result {
            eprintln!("wrdesk: {e}");
        }
        let _ = prompt("Press Enter to exit.");
        return Ok(());
    }
    result
}

fn join(target: Option<&str>, password: Option<String>, view_only: bool, interactive: bool) -> Result<()> {
    let (name, addr) = match resolve(target) {
        Err(e) if interactive => {
            println!("{e}");
            resolve(Some(&prompt("Host name or IP address: ")?))?
        }
        other => other?,
    };
    println!("Connecting to {name} ({})...", addr.ip());
    let password = match password {
        Some(p) => p,
        None => prompt("Password: ")?,
    };
    client::run(addr, &name, &password, view_only)
}

fn main() {
    if let Err(e) = run() {
        eprintln!("wrdesk: {e}");
        std::process::exit(1);
    }
}