//! System audio: the host records what its speakers play, the client plays it back.

use crate::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat, SizedSample, Stream, StreamConfig};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Records the default output device (loopback) until `alive` goes false or `send`
/// fails. `send` gets the sample rate, channel count and interleaved samples.
pub fn capture(alive: &AtomicBool, mut send: impl FnMut(u32, u16, &[i16]) -> Result<()>) -> Result<()> {
    let device = cpal::default_host().default_output_device().ok_or("no audio output device")?;
    let supported = device.default_output_config()?;
    let config = supported.config();
    let (tx, rx) = mpsc::channel();
    let stream = match supported.sample_format() {
        SampleFormat::F32 => record::<f32>(&device, &config, tx),
        SampleFormat::I16 => record::<i16>(&device, &config, tx),
        SampleFormat::I32 => record::<i32>(&device, &config, tx),
        format => return Err(format!("unsupported audio format {format:?}").into()),
    }?;
    stream.play()?;
    while alive.load(Ordering::Relaxed) {
        // Nothing arrives while nothing is playing.
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(chunk) => send(config.sample_rate, config.channels, &chunk)?,
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => return Err("audio capture stopped".into()),
        }
    }
    Ok(())
}

fn record<T>(device: &cpal::Device, config: &StreamConfig, tx: mpsc::Sender<Vec<i16>>) -> Result<Stream>
where
    T: SizedSample,
    i16: FromSample<T>,
{
    let data = move |data: &[T], _: &_| {
        let _ = tx.send(data.iter().map(|&s| i16::from_sample(s)).collect());
    };
    // Building an input stream on an output device records what it plays.
    Ok(device.build_input_stream(*config, data, |e| eprintln!("audio: {e}"), None)?)
}

/// Samples waiting to be played, already in the output device's rate and channel layout.
#[derive(Default)]
struct Queue {
    samples: VecDeque<f32>,
    playing: bool, // false while (re)filling after running dry
}

/// Plays the host's audio on the default output device.
pub struct Player {
    queue: Arc<Mutex<Queue>>,
    _stream: Stream,
    rate: u32,
    channels: usize,
    // Resampler state: the incoming format, the last incoming frame, and how far the
    // next output frame lies past it (in incoming frames).
    source: (u32, u16),
    prev: Vec<f32>,
    phase: f64,
}

impl Player {
    pub fn new() -> Result<Player> {
        let device = cpal::default_host().default_output_device().ok_or("no audio output device")?;
        let supported = device.default_output_config()?;
        let config = supported.config();
        let queue = Arc::new(Mutex::new(Queue::default()));
        let stream = match supported.sample_format() {
            SampleFormat::F32 => play::<f32>(&device, &config, queue.clone()),
            SampleFormat::I16 => play::<i16>(&device, &config, queue.clone()),
            SampleFormat::I32 => play::<i32>(&device, &config, queue.clone()),
            format => return Err(format!("unsupported audio format {format:?}").into()),
        }?;
        stream.play()?;
        Ok(Player {
            queue,
            _stream: stream,
            rate: config.sample_rate,
            channels: config.channels as usize,
            source: (0, 0),
            prev: Vec::new(),
            phase: 0.0,
        })
    }

    /// Queues interleaved samples recorded at `rate` with `channels` channels.
    pub fn push(&mut self, rate: u32, channels: u16, samples: &[i16]) {
        if rate == 0 || channels == 0 {
            return;
        }
        if self.source != (rate, channels) {
            self.source = (rate, channels);
            self.prev = vec![0.0; channels as usize];
            self.phase = 0.0;
        }
        let step = rate as f64 / self.rate as f64;
        let mut out = Vec::with_capacity((samples.len() as f64 / step) as usize + self.channels * 2);
        let mut cur = vec![0.0; channels as usize];
        for frame in samples.chunks_exact(channels as usize) {
            for (c, &s) in cur.iter_mut().zip(frame) {
                *c = s as f32 / 32768.0;
            }
            // Linear interpolation between the previous and current incoming frame.
            while self.phase < 1.0 {
                let t = self.phase as f32;
                for c in 0..self.channels {
                    let (a, b) = (mix(&self.prev, c, self.channels), mix(&cur, c, self.channels));
                    out.push(a + (b - a) * t);
                }
                self.phase += step;
            }
            self.phase -= 1.0;
            self.prev.copy_from_slice(&cur);
        }

        let frame_ms = self.rate as usize * self.channels / 1000;
        let mut queue = self.queue.lock().unwrap();
        queue.samples.extend(out);
        // Keep latency bounded when audio arrives faster than it plays (clock drift, bursts).
        if queue.samples.len() > 150 * frame_ms {
            let excess = queue.samples.len() - 50 * frame_ms;
            queue.samples.drain(..excess - excess % self.channels);
        }
    }
}

/// Output channel `c` (of `out` channels) from an incoming frame.
fn mix(frame: &[f32], c: usize, out: usize) -> f32 {
    match (frame.len(), out) {
        (1, _) => frame[0],
        (n, 1) => frame.iter().sum::<f32>() / n as f32,
        (n, _) if c < n => frame[c],
        _ => 0.0,
    }
}

fn play<T>(device: &cpal::Device, config: &StreamConfig, queue: Arc<Mutex<Queue>>) -> Result<Stream>
where
    T: SizedSample + FromSample<f32>,
{
    // Wait for this much audio before starting, to ride out network jitter.
    let prefill = config.sample_rate as usize * config.channels as usize * 40 / 1000;
    let data = move |out: &mut [T], _: &_| {
        let mut queue = queue.lock().unwrap();
        if !queue.playing && queue.samples.len() < prefill {
            return; // the buffer arrives filled with silence
        }
        queue.playing = true;
        for sample in out.iter_mut() {
            match queue.samples.pop_front() {
                Some(s) => *sample = T::from_sample(s),
                None => {
                    queue.playing = false;
                    break;
                }
            }
        }
    };
    Ok(device.build_output_stream(*config, data, |e| eprintln!("audio: {e}"), None)?)
}
