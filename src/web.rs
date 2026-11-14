//! Web client: the host serves a page that speaks the wire protocol over plain HTTP, so a
//! browser (e.g. on a phone) can view and control this machine. What the host sends is the
//! endless body of `GET /stream?id=..`; what the page sends arrives as `POST /input?id=..`
//! requests, one after the other. The id is a random number the page picks per connection.

use crate::host::{self, Link, Options};
use crate::{Result, WEB_PORT};
use qrcode::{Color, EcLevel, QrCode};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, ErrorKind, IsTerminal, Read, Write};
use std::net::{TcpListener, TcpStream, UdpSocket};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

const PAGE: &str = include_str!("web.html");

/// Where the input posted for each streaming browser goes, by connection id.
type Sessions = Arc<Mutex<HashMap<String, Sender<Vec<u8>>>>>;

/// Starts answering browsers in the background.
pub fn listen(password: &str, options: Options) -> Result<()> {
    let listener = TcpListener::bind(("0.0.0.0", WEB_PORT))?;
    // Connecting a UDP socket sends nothing; it only picks the address this machine has on the LAN.
    let lan = UdpSocket::bind(("0.0.0.0", 0)).and_then(|socket| {
        socket.connect(("8.8.8.8", 53))?;
        socket.local_addr()
    });
    match lan {
        Ok(addr) => {
            let url = format!("http://{}:{WEB_PORT}", addr.ip());
            println!("From a phone or browser, open {url}");
            if std::io::stdout().is_terminal() {
                print_qr(&url);
            }
        }
        Err(_) => println!("From a phone or browser, open http://<this computer's IP address>:{WEB_PORT}"),
    }
    let password = password.to_string();
    let sessions = Sessions::default();
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { continue };
            let (password, sessions) = (password.clone(), sessions.clone());
            thread::spawn(move || {
                let peer = stream.peer_addr().map(|a| a.ip().to_string()).unwrap_or_default();
                if let Err(e) = handle(stream, &password, options, &sessions) {
                    println!("{peer}: web request failed: {e}");
                }
            });
        }
    });
    Ok(())
}

/// Prints `text` as a QR code for a phone camera. Each line of text shows two rows of
/// the code, drawn black on white whatever the terminal's colours are.
fn print_qr(text: &str) {
    let Ok(code) = QrCode::with_error_correction_level(text, EcLevel::L) else { return };
    const MARGIN: i32 = 2;
    let size = code.width() as i32;
    let dark = |x: i32, y: i32| {
        (0..size).contains(&x) && (0..size).contains(&y) && code[(x as usize, y as usize)] == Color::Dark
    };
    for y in (-MARGIN..size + MARGIN).step_by(2) {
        let row: String = (-MARGIN..size + MARGIN)
            .map(|x| match (dark(x, y), dark(x, y + 1)) {
                (true, true) => '█',
                (true, false) => '▀',
                (false, true) => '▄',
                (false, false) => ' ',
            })
            .collect();
        println!("  \x1b[30;107m{row}\x1b[0m");
    }
}

/// Answers one HTTP request: with the page, by streaming a session, or by taking input for one.
fn handle(mut stream: TcpStream, password: &str, options: Options, sessions: &Sessions) -> Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(10)))?;
    let mut reader = BufReader::new(stream.try_clone()?.take(16 << 10));
    let mut line = String::new();
    // Browsers open spare connections they never use: those end here, quietly.
    if !matches!(reader.read_line(&mut line), Ok(n) if n > 0) {
        return Ok(());
    }
    let mut request = line.split_whitespace().map(str::to_string);
    let (method, target) = (request.next().unwrap_or_default(), request.next().unwrap_or_default());
    let (path, id) = target.split_once("?id=").unwrap_or((&target, ""));
    let (mut host, mut origin, mut length) = (String::new(), None, 0);
    loop {
        line.clear();
        reader.read_line(&mut line)?;
        let Some((name, value)) = line.split_once(':') else { break };
        let value = value.trim();
        match name.to_ascii_lowercase().as_str() {
            "host" => host = value.to_string(),
            "origin" => origin = Some(value.to_string()),
            "content-length" => length = value.parse()?,
            _ => {}
        }
    }

    let reply = |stream: &mut TcpStream, status: &str, kind: &str, body: &str| {
        write!(
            stream,
            "HTTP/1.1 {status}\r\nContent-Type: {kind}\r\nContent-Length: {}\r\n\
             Cache-Control: no-store\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
    };
    match (method.as_str(), path) {
        ("GET", "/") => {
            let page = PAGE.replace("__LOCKED__", if password.is_empty() { "false" } else { "true" });
            reply(&mut stream, "200 OK", "text/html; charset=utf-8", &page)?;
        }
        ("GET", "/stream") if !id.is_empty() => {
            // No length: the body is the session, and ends when the connection does.
            stream.write_all(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nX-Content-Type-Options: nosniff\r\n\
                  Cache-Control: no-store\r\nConnection: close\r\n\r\n",
            )?;
            let (sender, posts) = mpsc::channel();
            sessions.lock().unwrap().insert(id.to_string(), sender);
            host::session(Browser { tcp: stream, input: Arc::new(Mutex::new((posts, Vec::new()))) }, password, options);
            sessions.lock().unwrap().remove(id); // also ends the session's input thread
        }
        ("POST", "/input") => {
            let mut body = vec![0; length];
            reader.read_exact(&mut body)?;
            // Browsers let any website post to us; only our own page may.
            let ours = origin.is_none_or(|origin| origin.strip_prefix("http://") == Some(host.as_str()));
            let session = sessions.lock().unwrap().get(id).cloned();
            let taken = ours && session.is_some_and(|session| session.send(body).is_ok());
            reply(&mut stream, if taken { "204 No Content" } else { "410 Gone" }, "text/plain", "")?;
        }
        _ => reply(&mut stream, "404 Not Found", "text/plain", "")?,
    }
    Ok(())
}

/// A browser as a byte stream: writes go out as the body of its `/stream` request, and
/// reads return what it posts.
struct Browser {
    tcp: TcpStream,
    input: Arc<Mutex<(Receiver<Vec<u8>>, Vec<u8>)>>, // the posts, and what is unread of the last one
}

impl Link for Browser {
    fn socket(&self) -> &TcpStream {
        &self.tcp
    }
    fn try_clone(&self) -> std::io::Result<Self> {
        Ok(Browser { tcp: self.tcp.try_clone()?, input: self.input.clone() })
    }
}

impl Read for Browser {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let mut input = self.input.lock().unwrap();
        let (posts, unread) = &mut *input;
        while unread.is_empty() {
            // The socket's read timeout is the one that is meant for us.
            let post = match self.tcp.read_timeout()? {
                Some(limit) => posts.recv_timeout(limit),
                None => posts.recv().map_err(|_| RecvTimeoutError::Disconnected),
            };
            match post {
                Ok(bytes) => *unread = bytes,
                Err(RecvTimeoutError::Timeout) => return Err(ErrorKind::TimedOut.into()),
                Err(RecvTimeoutError::Disconnected) => return Ok(0), // the session is over
            }
        }
        let n = buf.len().min(unread.len());
        buf[..n].copy_from_slice(&unread[..n]);
        unread.drain(..n);
        Ok(n)
    }
}

impl Write for Browser {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.tcp.write(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.tcp.flush()
    }
}
