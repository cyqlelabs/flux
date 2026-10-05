//! flux-worker: one process per engine instance. Owns the model and devices for a session and
//! speaks the versioned protocol on stdio; also runs one-shot native jobs (measure, probes)
//! so that native failures never take down the supervisor.

mod external;
mod native;
mod out;
mod text;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use flux_core::protocol::{Event, Request, PROTOCOL_VERSION};
use std::io::{BufRead, Read};

#[derive(Parser)]
#[command(version, about = "Flux engine worker (spawned by flux; speaks JSON lines on stdio)")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Print the pinned backend's build and devices.
    Info,
    /// Allocation dry run for BackendParams JSON on stdin.
    Measure,
    /// Run a native probe; request JSON on stdin.
    Probe { kind: ProbeKind },
    /// Serve the protocol with the native llama.cpp engine.
    Serve,
    /// Serve the protocol by driving an engine that has its own HTTP server.
    External {
        #[arg(long, value_enum)]
        api: ApiArg,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum ProbeKind {
    Matmul,
    Copy,
    Contention,
    HostPages,
    Supports,
}

#[derive(Clone, Copy, ValueEnum)]
enum ApiArg {
    LlamaServer,
    OpenaiChat,
}

pub fn rss_bytes() -> u64 {
    let statm = std::fs::read_to_string("/proc/self/statm").unwrap_or_default();
    let pages: u64 = statm.split_whitespace().nth(1).and_then(|v| v.parse().ok()).unwrap_or(0);
    pages * unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64
}

fn stdin_json() -> Result<serde_json::Value> {
    let mut s = String::new();
    std::io::stdin().read_to_string(&mut s)?;
    Ok(serde_json::from_str(&s)?)
}

fn print_result(r: Result<serde_json::Value>) {
    let v = r.unwrap_or_else(|e| serde_json::json!({"error": format!("{e:#}")}));
    println!("{v}");
}

fn hello(out: &out::Out, engine: &str, level: &str) {
    out.send(&Event::Hello {
        protocol: PROTOCOL_VERSION,
        worker: env!("CARGO_PKG_VERSION").into(),
        engine: engine.into(),
        backend_revision: flux_native::BACKEND_PIN.into(),
        backend_build: flux_native::BACKEND_BUILD.into(),
        level: level.into(),
    });
}

fn main() -> Result<()> {
    flux_native::verify_libraries()?;
    match Cli::parse().cmd {
        Cmd::Info => print_result(flux_native::backend_info().map(|mut v| {
            v["pin"] = flux_native::BACKEND_PIN.into();
            v["build"] = flux_native::BACKEND_BUILD.into();
            v
        })),
        Cmd::Measure => print_result(stdin_json().and_then(|p| flux_native::measure(&p))),
        Cmd::Probe { kind } => {
            let f = match kind {
                ProbeKind::Matmul => flux_native::probe_matmul,
                ProbeKind::Copy => flux_native::probe_copy,
                ProbeKind::Contention => flux_native::probe_contention,
                ProbeKind::HostPages => flux_native::probe_host_pages,
                ProbeKind::Supports => flux_native::supports,
            };
            print_result(stdin_json().and_then(|r| f(&r)));
        }
        Cmd::Serve => {
            let out = out::Out::stdout();
            hello(&out, "native", "tokens");
            let (tx, rx) = std::sync::mpsc::sync_channel(64);
            let reader_out = out.clone();
            std::thread::spawn(move || {
                for line in std::io::stdin().lock().lines() {
                    let Ok(line) = line else { break };
                    match serde_json::from_str::<Request>(&line) {
                        Ok(r) => {
                            if tx.send(r).is_err() {
                                return;
                            }
                        }
                        Err(e) => reader_out.send(&Event::Error {
                            req: None,
                            id: None,
                            code: flux_core::protocol::ErrorCode::Protocol,
                            message: format!("bad request line: {e}"),
                        }),
                    }
                }
                let _ = tx.send(Request::Shutdown);
            });
            native::NativeWorker::new(out).run(rx)?;
        }
        Cmd::External { api } => {
            let out = out::Out::stdout();
            let (api, name, level) = match api {
                ApiArg::LlamaServer => (external::Api::LlamaServer, "llama-server", "tokens"),
                ApiArg::OpenaiChat => (external::Api::OpenAiChat, "external", "chat"),
            };
            hello(&out, name, level);
            tokio::runtime::Runtime::new()?.block_on(external::run(out, api))?;
        }
    }
    Ok(())
}
