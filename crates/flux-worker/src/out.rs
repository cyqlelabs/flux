use flux_core::protocol::Event;
use std::io::Write;
use std::sync::{Arc, Mutex};

/// Serialized writer for protocol events: one JSON object per line on stdout.
#[derive(Clone)]
pub struct Out {
    inner: Arc<Mutex<std::io::Stdout>>,
}

impl Out {
    pub fn stdout() -> Out {
        Out { inner: Arc::new(Mutex::new(std::io::stdout())) }
    }

    pub fn send(&self, ev: &Event) {
        let mut line = serde_json::to_vec(ev).expect("event serializes");
        line.push(b'\n');
        let mut o = self.inner.lock().expect("stdout lock");
        // A closed pipe means the supervisor is gone; the read loop will see EOF and exit.
        let _ = o.write_all(&line).and_then(|_| o.flush());
    }
}
