//! Token bytes → client text: completes UTF-8 sequences split across tokens and
//! holds back text that could still turn into a stop string.

pub enum Pushed {
    Text(String),
    /// A stop string matched; carries the text before it. Generation ends here.
    Stop(String),
}

pub struct TextStream {
    utf8: Vec<u8>,
    held: String,
    stops: Vec<String>,
}

impl TextStream {
    pub fn new(stops: Vec<String>) -> TextStream {
        TextStream { utf8: vec![], held: String::new(), stops: stops.into_iter().filter(|s| !s.is_empty()).collect() }
    }

    pub fn push(&mut self, bytes: &[u8]) -> Pushed {
        self.utf8.extend_from_slice(bytes);
        self.decode_valid();
        if let Some(i) = self.stops.iter().filter_map(|s| self.held.find(s.as_str())).min() {
            let before = self.held[..i].to_string();
            self.held.clear();
            self.utf8.clear();
            return Pushed::Stop(before);
        }
        let keep = self.partial_stop_suffix();
        let emit = self.held[..self.held.len() - keep].to_string();
        self.held.drain(..self.held.len() - keep);
        Pushed::Text(emit)
    }

    /// Everything still held, with incomplete UTF-8 replaced.
    pub fn finish(&mut self) -> String {
        let mut out = std::mem::take(&mut self.held);
        out.push_str(&String::from_utf8_lossy(&std::mem::take(&mut self.utf8)));
        out
    }

    fn decode_valid(&mut self) {
        loop {
            match std::str::from_utf8(&self.utf8) {
                Ok(s) => {
                    self.held.push_str(s);
                    self.utf8.clear();
                    return;
                }
                Err(e) => {
                    let ok = e.valid_up_to();
                    self.held.push_str(std::str::from_utf8(&self.utf8[..ok]).expect("valid prefix"));
                    match e.error_len() {
                        // Incomplete sequence at the end: wait for the next token.
                        None => {
                            self.utf8.drain(..ok);
                            return;
                        }
                        Some(bad) => {
                            self.held.push(char::REPLACEMENT_CHARACTER);
                            self.utf8.drain(..ok + bad);
                        }
                    }
                }
            }
        }
    }

    /// Length of the longest suffix of `held` that is a proper prefix of some stop string.
    fn partial_stop_suffix(&self) -> usize {
        let mut best = 0;
        for s in &self.stops {
            for (i, _) in self.held.char_indices() {
                let suffix = &self.held[i..];
                if suffix.len() < s.len() && s.starts_with(suffix) {
                    best = best.max(suffix.len());
                    break;
                }
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(p: Pushed) -> String {
        match p {
            Pushed::Text(t) => t,
            Pushed::Stop(_) => panic!("unexpected stop"),
        }
    }

    #[test]
    fn joins_split_utf8() {
        let mut t = TextStream::new(vec![]);
        let euro = "€".as_bytes();
        assert_eq!(text(t.push(&euro[..1])), "");
        assert_eq!(text(t.push(&euro[1..])), "€");
    }

    #[test]
    fn holds_partial_stop_and_cuts_at_match() {
        let mut t = TextStream::new(vec!["</end>".into()]);
        assert_eq!(text(t.push(b"hello </")), "hello ");
        match t.push(b"end> tail") {
            Pushed::Stop(before) => assert_eq!(before, ""),
            Pushed::Text(_) => panic!("stop missed"),
        }
    }

    #[test]
    fn releases_held_text_when_no_match() {
        let mut t = TextStream::new(vec!["</end>".into()]);
        assert_eq!(text(t.push(b"a </e")), "a ");
        assert_eq!(text(t.push(b"x")), "</ex");
        assert_eq!(text(t.push(b" </")), " ");
        assert_eq!(t.finish(), "</");
    }

    #[test]
    fn invalid_bytes_become_replacement() {
        let mut t = TextStream::new(vec![]);
        assert_eq!(text(t.push(&[0xff, b'a'])), "\u{fffd}a");
    }
}
