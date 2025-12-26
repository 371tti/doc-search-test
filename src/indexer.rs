use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use half::f16;
use indicatif::{HumanBytes, ProgressBar, ProgressDrawTarget, ProgressStyle};
use jwalk::WalkDir;
use tf_idf_vectorizer::{Corpus, Hits, Query, SimilarityAlgorithm, TFIDFData, TFIDFVectorizer, TokenFrequency};

use crate::tokenize::SudachiTokenizer;

pub type DynError = Box<dyn std::error::Error + Send + Sync>;

pub struct DocIndexer {
    pub corpus: Arc<Corpus>,
    pub vectorizer: TFIDFVectorizer<f16>,
    pub tokenizer: SudachiTokenizer,
}

const QUEUE_CAPACITY: usize = 1024;

pub struct Queues {
    pub file_reader_sender: flume::Sender<PathBuf>,
    pub file_reader_receiver: flume::Receiver<PathBuf>,
    pub tokenize_sender: flume::Sender<WithKey<Vec<u8>>>,
    pub tokenize_receiver: flume::Receiver<WithKey<Vec<u8>>>,
    pub vectorize_sender: flume::Sender<WithKey<TokenFrequency>>,
    pub vectorize_receiver: flume::Receiver<WithKey<TokenFrequency>>,
}

#[derive(Default)]
pub struct PipelineStats {
    // path stage
    pub paths_sent: AtomicU64,

    // read stage
    pub read_committed: AtomicU64,
    pub read_ok: AtomicU64,
    pub read_err: AtomicU64,
    pub utf8_err: AtomicU64,
    pub bytes_read: AtomicU64,

    // tokenize stage
    pub tokenize_committed: AtomicU64,
    pub tok_ok: AtomicU64,
    pub tok_err: AtomicU64,

    // index stage
    pub index_committed: AtomicU64,
    pub docs_added: AtomicU64,
}

pub struct WithKey<T> {
    pub key: String,
    pub value: T,
}

impl Queues {
    pub fn new() -> Self {
        let (file_reader_sender, file_reader_receiver) = flume::bounded(QUEUE_CAPACITY);
        let (tokenize_sender, tokenize_receiver) = flume::bounded(QUEUE_CAPACITY);
        let (vectorize_sender, vectorize_receiver) = flume::bounded(QUEUE_CAPACITY);
        Self {
            file_reader_sender,
            file_reader_receiver,
            tokenize_sender,
            tokenize_receiver,
            vectorize_sender,
            vectorize_receiver,
        }
    }
}

struct ProgressReader<R> {
    inner: R,
    pb: ProgressBar,
}

impl<R: Read> Read for ProgressReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        if n > 0 {
            self.pb.inc(n as u64);
        }
        Ok(n)
    }
}

struct ProgressWriter<W> {
    inner: W,
    pb: ProgressBar,
    written: u64,
}

impl<W: Write> Write for ProgressWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        if n > 0 {
            self.written = self.written.saturating_add(n as u64);
            self.pb.inc(n as u64);
            if self.written % (4 * 1024 * 1024) < n as u64 {
                self.pb.set_message(format!("{}", HumanBytes(self.written)));
            }
        }
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

fn is_text_allow(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|s| s.to_str()).map(|s| s.to_ascii_lowercase()),
        Some(ext) if matches!(
            ext.as_str(),
            "txt" | "md" | "rs" | "c" | "cpp" | "h" |
            "py" | "js" | "ts" | "json" | "toml" |
            "yaml" | "yml" | "html" | "css"
        )
    )
}

impl DocIndexer {
    pub fn new() -> Self {
        let corpus = Arc::new(Corpus::new());
        let vectorizer = TFIDFVectorizer::new(corpus.clone());
        let tokenizer = SudachiTokenizer::new().unwrap();
        DocIndexer { corpus, vectorizer, tokenizer }
    }

    pub fn spawn_path_collector_thread(root_dir: PathBuf, queues: &Queues, stats: Arc<PipelineStats>) -> JoinHandle<()> {
        let sender = queues.file_reader_sender.clone();
        std::thread::spawn(move || {
            for e in jwalk::WalkDir::new(root_dir).into_iter().filter_map(Result::ok) {
                if e.file_type().is_file() {
                    let path = e.path();
                    if is_text_allow(&path) {
                        if sender.send(path).is_err() { return; }
                        stats.paths_sent.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        })
    }

    pub fn spawm_file_reader_thread(queues: &Queues, stats: Arc<PipelineStats>) -> JoinHandle<()> {
        let receiver = queues.file_reader_receiver.clone();
        let sender = queues.tokenize_sender.clone();
        std::thread::spawn(move || {
            while let Ok(path) = receiver.recv() {
                stats.read_committed.fetch_add(1, Ordering::Relaxed);
                let key = path
                    .file_name()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| path.to_string_lossy().into_owned());

                let bytes = match std::fs::read(&path) {
                    Ok(b) => b,
                    Err(_) => { stats.read_err.fetch_add(1, Ordering::Relaxed); continue; }
                };
                stats.bytes_read.fetch_add(bytes.len() as u64, Ordering::Relaxed);

                if sender.send(WithKey { key, value: bytes }).is_err() {
                    return;
                }
                stats.read_ok.fetch_add(1, Ordering::Relaxed);
            }
        })
    }

    pub fn spawn_tokenize_thread(&self, queues: &Queues, stats: Arc<PipelineStats>) -> JoinHandle<()> {
        let receiver = queues.tokenize_receiver.clone();
        let sender = queues.vectorize_sender.clone();
        let tokenizer = self.tokenizer.clone();
        std::thread::spawn(move || {
            while let Ok(text) = receiver.recv() {
                stats.tokenize_committed.fetch_add(1, Ordering::Relaxed);
                let s = match std::str::from_utf8(&text.value) {
                    Ok(s) => s,
                    Err(_) => { stats.utf8_err.fetch_add(1, Ordering::Relaxed); continue; }
                };
                let tokenized = match tokenizer.mix_doc_tokenizer(&s) {
                    Ok(t) => t,
                    Err(_) => { stats.tok_err.fetch_add(1, Ordering::Relaxed); continue; }
                };
                let freq = TokenFrequency::from(tokenized.0.as_slice());
                if sender.send(WithKey { key: text.key, value: freq }).is_err() {
                    return;
                }
                stats.tok_ok.fetch_add(1, Ordering::Relaxed);
            }
        })
    }
    
    fn spawn_metrics_thread(stats: Arc<PipelineStats>, total_files: u64) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || {
            let pb = ProgressBar::new(total_files);
            pb.set_draw_target(ProgressDrawTarget::stderr_with_hz(10));
            pb.set_style(
                ProgressStyle::with_template(
                    "{spinner:.green} [{elapsed_precise}] {wide_bar:.cyan/blue} {pos}/{len} ({percent}%) \
                    idx {per_sec:0} ETA {eta_precise}\n{msg}"
                )
                .unwrap()
                .progress_chars("#>-")
                .tick_strings(&[
                    "|", "/", "-", "\\",
                ]),
            );
            pb.enable_steady_tick(Duration::from_millis(120));

            let mut last = Instant::now();
            let mut last_read_ok = 0u64;
            let mut last_tok_ok  = 0u64;
            let mut last_added   = 0u64;
            let mut last_bytes   = 0u64;

            loop {
                std::thread::sleep(Duration::from_millis(200));

                let read_ok = stats.read_ok.load(Ordering::Relaxed);
                let tok_ok  = stats.tok_ok.load(Ordering::Relaxed);
                let added   = stats.docs_added.load(Ordering::Relaxed);
                let bytes   = stats.bytes_read.load(Ordering::Relaxed);

                let read_err = stats.read_err.load(Ordering::Relaxed);
                let utf8_err = stats.utf8_err.load(Ordering::Relaxed);
                let tok_err  = stats.tok_err.load(Ordering::Relaxed);

                let read_committed = stats.read_committed.load(Ordering::Relaxed);
                let tokenize_committed = stats.tokenize_committed.load(Ordering::Relaxed);
                let index_committed = stats.index_committed.load(Ordering::Relaxed);

                let paths_sent = stats.paths_sent.load(Ordering::Relaxed);

                // progress は “最終成果物” (= added) に合わせるのが正確
                pb.set_position(added.min(total_files));

                let dt = last.elapsed().as_secs_f64().max(1e-9);
                let read_rps = (read_ok - last_read_ok) as f64 / dt;
                let tok_rps  = (tok_ok  - last_tok_ok ) as f64 / dt;
                let add_rps  = (added   - last_added  ) as f64 / dt;
                let mbps     = (bytes   - last_bytes  ) as f64 / dt / (1024.0 * 1024.0);

                last = Instant::now();
                last_read_ok = read_ok;
                last_tok_ok  = tok_ok;
                last_added   = added;
                last_bytes   = bytes;

                let read_queue_len = paths_sent.saturating_sub(read_committed);
                let tok_queue_len = (read_ok).saturating_sub(tokenize_committed);
                let index_queue_len = (tok_ok).saturating_sub(index_committed);
                let seen = paths_sent;

                pb.set_message(format!(
                    "io:    {:>8.1} MiB/s    (total {:.1} GiB)\n\
                    ↓ read:  {:>8.1} files/s  ({read_ok} ok, {read_err} err, {utf8_err} utf8, {read_queue_len} wait)\n\
                    ↓ tok:   {:>8.1} docs/s   ({tok_ok} ok, {tok_err} err, {tok_queue_len} wait)\n\
                    ↓ index: {:>8.1} docs/s   ({added} added, {index_queue_len} wait)\n\
                    Ram\n",
                    mbps,
                    bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                    read_rps,
                    tok_rps,
                    add_rps,
                ));

                // すべて追加し終わったら終了
                if seen >= total_files as u64 && index_queue_len == 0 {
                    pb.finish_with_message("done");
                    break;
                }
            }
        })
    }

    pub fn build_index(&mut self, path: &Path) -> Result<(), DynError> {


        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} {pos} files | {elapsed_precise} | {per_sec} files/s"
            ).unwrap()
        );
        pb.enable_steady_tick(std::time::Duration::from_millis(100));

        let mut file_sum = 0usize;

        for e in WalkDir::new(path)
            .into_iter()
            .filter_map(Result::ok)
        {
            if e.file_type().is_file() && is_text_allow(&e.path()) {
                file_sum += 1;
                pb.inc(1);
            }
        }

        pb.finish_with_message(format!("done: {} files", file_sum));

        let queues = Queues::new();
        let stats = Arc::new(PipelineStats::default());
        let metrics_handle = Self::spawn_metrics_thread(stats.clone(), file_sum as u64);

        let path_collector_handle = DocIndexer::spawn_path_collector_thread(path.to_path_buf(), &queues, stats.clone());

        // ファイルリーダースレッドを複数立てる
        let file_reader_threads = 6;
        let mut file_reader_handles = Vec::with_capacity(file_reader_threads);
        for _ in 0..file_reader_threads {
            file_reader_handles.push(DocIndexer::spawm_file_reader_thread(&queues, stats.clone()));
        }

        // トークナイズスレッドを複数立てる
        let tokenize_threads = 32;
        let mut tokenize_handles = Vec::with_capacity(tokenize_threads);
        for _ in 0..tokenize_threads {
            tokenize_handles.push(self.spawn_tokenize_thread(&queues, stats.clone()));
        }

        drop(queues.file_reader_sender);
        drop(queues.tokenize_sender);
        drop(queues.vectorize_sender);

        for WithKey { key, value: freq } in queues.vectorize_receiver.iter() {
            stats.index_committed.fetch_add(1, Ordering::Relaxed);
            self.vectorizer.add_doc(key, &freq);
            stats.docs_added.fetch_add(1, Ordering::Relaxed);
        }

        path_collector_handle.join().unwrap();
        for handle in file_reader_handles {
            handle.join().unwrap();
        }
        for handle in tokenize_handles {
            handle.join().unwrap();
        }
        metrics_handle.join().unwrap();

        Ok(())
    }

    pub fn save_to(&self, index_dir: &Path) -> Result<(), DynError> {
        std::fs::create_dir_all(index_dir)?;

        let corpus_file = std::fs::File::create(index_dir.join("corpus.cbor"))?;
        let vector_file = std::fs::File::create(index_dir.join("vector.cbor"))?;

        let pb_corpus = ProgressBar::new_spinner();
        pb_corpus.set_draw_target(ProgressDrawTarget::stderr_with_hz(10));
        pb_corpus.enable_steady_tick(std::time::Duration::from_millis(120));
        pb_corpus.set_style(
            ProgressStyle::with_template("{spinner:.green} saving corpus... {msg}")
                .unwrap()
                .progress_chars("##-"),
        );

        let pb_vector = ProgressBar::new_spinner();
        pb_vector.set_draw_target(ProgressDrawTarget::stderr_with_hz(10));
        pb_vector.enable_steady_tick(std::time::Duration::from_millis(120));
        pb_vector.set_style(
            ProgressStyle::with_template("{spinner:.green} saving vector... {msg}")
                .unwrap()
                .progress_chars("#>-"),
        );

        let corpus_writer = BufWriter::new(corpus_file);
        let vector_writer = BufWriter::new(vector_file);

        let mut corpus_writer = ProgressWriter {
            inner: corpus_writer,
            pb: pb_corpus.clone(),
            written: 0,
        };
        let mut vector_writer = ProgressWriter {
            inner: vector_writer,
            pb: pb_vector.clone(),
            written: 0,
        };

        ciborium::into_writer(&*self.corpus, &mut corpus_writer)?;
        corpus_writer.flush()?;
        pb_corpus.finish_with_message(format!("{}", HumanBytes(corpus_writer.written)));

        ciborium::into_writer(&self.vectorizer, &mut vector_writer)?;
        vector_writer.flush()?;
        pb_vector.finish_with_message(format!("{}", HumanBytes(vector_writer.written)));

        pb_corpus.finish_and_clear();
        pb_vector.finish_and_clear();
        Ok(())
    }

    pub fn load_from(index_dir: &Path) -> Result<Self, DynError> {
        let inst = std::time::Instant::now();
        let corpus_path = index_dir.join("corpus.cbor");
        let vector_path = index_dir.join("vector.cbor");

        let corpus_file = std::fs::File::open(&corpus_path)?;
        let vector_file = std::fs::File::open(&vector_path)?;

        let corpus_len = std::fs::metadata(&corpus_path).map(|m| m.len()).unwrap_or(0);
        let vector_len = std::fs::metadata(&vector_path).map(|m| m.len()).unwrap_or(0);

        let pb_corpus = ProgressBar::new(corpus_len);
        pb_corpus.set_draw_target(ProgressDrawTarget::stderr_with_hz(10));
        pb_corpus.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} loading corpus... {wide_bar:.cyan/blue} {bytes}/{total_bytes} ({percent}%) ETA {eta_precise}",
            )
            .unwrap()
            .progress_chars("#>-"),
        );

        let pb_vector = ProgressBar::new(vector_len);
        pb_vector.set_draw_target(ProgressDrawTarget::stderr_with_hz(10));
        pb_vector.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} loading vector... {wide_bar:.cyan/blue} {bytes}/{total_bytes} ({percent}%) ETA {eta_precise}",
            )
            .unwrap()
            .progress_chars("#>-"),
        );

        let corpus_reader = BufReader::new(corpus_file);
        let vector_reader = BufReader::new(vector_file);

        let corpus_reader = ProgressReader {
            inner: corpus_reader,
            pb: pb_corpus.clone(),
        };
        let vector_reader = ProgressReader {
            inner: vector_reader,
            pb: pb_vector.clone(),
        };

        let corpus: Corpus = ciborium::from_reader(corpus_reader)?;
        pb_corpus.finish();
        let corpus = Arc::new(corpus);

        let vectorizer: TFIDFData<f16> = ciborium::from_reader(vector_reader)?;
        pb_vector.finish();
        println!("Extending vectorizer...");
        let vectorizer = vectorizer.into_tf_idf_vectorizer(corpus.clone());
        let doc_num = vectorizer.doc_num();
        let vocab_size = corpus.vocab_size();
        let token_sample_dim_size = vectorizer.token_dim_rev_index.len();
        let max_rev_idx_len = vectorizer.token_dim_rev_index.values().iter().map(|v| v.len()).max().unwrap_or(0);
        let tokenizer = SudachiTokenizer::new()?;
        let elapsed = inst.elapsed().as_millis();
        println!("{} documents loaded. Vocab size: {}. Token sample dim size: {}. Max rev idx len: {}. Done {} ms", doc_num, vocab_size, token_sample_dim_size, max_rev_idx_len, elapsed);
        Ok(DocIndexer { corpus, vectorizer, tokenizer })
    }

    /// # Query format
    /// key_word:
    /// - &: AND
    /// - |: OR
    /// - !: NOT
    /// - []: group
    /// 
    /// example:
    /// "[rust & tf-idf | tokenizer] & !game"
    /// to
    /// let query = Query::and(Query::or(Query::and(Query::token("rust"), Query::token("tf-idf")), Query::token("tokenizer")), Query::not(Query::token("game")));
    pub fn search_doc(&mut self, query: &str, algorithm: SimilarityAlgorithm) -> Result<Hits<String>, DynError> {
        let inst = std::time::Instant::now();

        let query = QueryBuilder::new(query).build()?;

        let mut results = self.vectorizer.search(&algorithm, query);
        results.sort_by_score_desc();
        let elapsed = inst.elapsed().as_millis();
        println!("Found {} results in {} ms.", results.list.len(), elapsed);
        Ok(results)
    }


}
#[derive(Debug, Clone)]
pub struct QueryBuilder {
    tokens: Vec<QueryToken>,
    pos: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum QueryToken {
    LBracket,
    RBracket,
    And,
    Or,
    Not,
    Word(String),
}

#[derive(Debug)]
struct QueryParseError(String);

impl std::fmt::Display for QueryParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Query parse error: {}", self.0)
    }
}

impl std::error::Error for QueryParseError {}

impl QueryBuilder {
    pub fn new(query: &str) -> Self {
        Self {
            tokens: Self::tokenize(query),
            pos: 0,
        }
    }

    /// Parse the query string into `tf_idf_vectorizer::Query`.
    ///
    /// Supported:
    /// - `&` AND
    /// - `|` OR
    /// - `!` NOT (prefix)
    /// - `[...]` grouping
    ///
    /// Also supported:
    /// - whitespace between terms is treated as AND
    ///   e.g. `[token0 token1]` == `[token0 & token1]`
    ///
    /// Precedence: `!` > (implicit/explicit `&`) > `|`
    pub fn build(mut self) -> Result<Query, DynError> {
        if self.tokens.is_empty() {
            return Ok(Query::none());
        }

        let q = self.parse_or()?;

        if self.pos != self.tokens.len() {
            return Err(Box::new(QueryParseError(format!(
                "unexpected token at end: {:?}",
                self.tokens.get(self.pos)
            ))));
        }

        Ok(q)
    }

    fn tokenize(query: &str) -> Vec<QueryToken> {
        let mut out = Vec::new();
        let mut buf = String::new();

        let flush_word = |out: &mut Vec<QueryToken>, buf: &mut String| {
            if !buf.is_empty() {
                out.push(QueryToken::Word(std::mem::take(buf)));
            }
        };

        for ch in query.chars() {
            match ch {
                // whitespace ends a word (implicit AND is handled in parsing)
                c if c.is_whitespace() => {
                    flush_word(&mut out, &mut buf);
                }
                '[' => {
                    flush_word(&mut out, &mut buf);
                    out.push(QueryToken::LBracket);
                }
                ']' => {
                    flush_word(&mut out, &mut buf);
                    out.push(QueryToken::RBracket);
                }
                '&' => {
                    flush_word(&mut out, &mut buf);
                    out.push(QueryToken::And);
                }
                '|' => {
                    flush_word(&mut out, &mut buf);
                    out.push(QueryToken::Or);
                }
                '!' => {
                    flush_word(&mut out, &mut buf);
                    out.push(QueryToken::Not);
                }
                _ => buf.push(ch),
            }
        }

        flush_word(&mut out, &mut buf);
        out
    }

    fn peek(&self) -> Option<&QueryToken> {
        self.tokens.get(self.pos)
    }

    fn next(&mut self) -> Option<QueryToken> {
        let t = self.tokens.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    fn expect(&mut self, want: QueryToken) -> Result<(), DynError> {
        let got = self.next();
        if got.as_ref() == Some(&want) {
            Ok(())
        } else {
            Err(Box::new(QueryParseError(format!(
                "expected {:?}, got {:?}",
                want, got
            ))))
        }
    }

    // Lowest precedence
    fn parse_or(&mut self) -> Result<Query, DynError> {
        let mut left = self.parse_and()?;
        while matches!(self.peek(), Some(QueryToken::Or)) {
            self.next();
            let right = self.parse_and()?;
            left = Query::or(left, right);
        }
        Ok(left)
    }

    // AND precedence (explicit '&' or implicit by whitespace / adjacency)
    fn parse_and(&mut self) -> Result<Query, DynError> {
        let mut left = self.parse_not()?;

        loop {
            match self.peek() {
                Some(QueryToken::And) => {
                    self.next(); // explicit AND
                    let right = self.parse_not()?;
                    left = Query::and(left, right);
                }
                // implicit AND: next token starts an expression
                Some(QueryToken::Word(_)) | Some(QueryToken::LBracket) | Some(QueryToken::Not) => {
                    let right = self.parse_not()?;
                    left = Query::and(left, right);
                }
                _ => break,
            }
        }

        Ok(left)
    }

    // Highest precedence (prefix unary)
    fn parse_not(&mut self) -> Result<Query, DynError> {
        if matches!(self.peek(), Some(QueryToken::Not)) {
            self.next();
            let inner = self.parse_not()?;
            Ok(Query::not(inner))
        } else {
            self.parse_primary()
        }
    }

    fn parse_primary(&mut self) -> Result<Query, DynError> {
        match self.next() {
            Some(QueryToken::Word(mut w)) => {
                // allow quoted tokens: "token" or 'token'
                if w.len() >= 2 {
                    let bytes = w.as_bytes();
                    let first = bytes[0];
                    let last = bytes[bytes.len() - 1];
                    if (first == b'"' && last == b'"') || (first == b'\'' && last == b'\'') {
                        w = w[1..w.len() - 1].to_string();
                    }
                }

                if w.is_empty() {
                    Ok(Query::none())
                } else if w == "*" {
                    Ok(Query::all())
                } else {
                    Ok(Query::token(&w))
                }
            }
            Some(QueryToken::LBracket) => {
                let inner = self.parse_or()?;
                self.expect(QueryToken::RBracket)?;
                Ok(inner)
            }
            Some(tok) => Err(Box::new(QueryParseError(format!(
                "unexpected token: {:?}",
                tok
            )))),
            None => Ok(Query::none()),
        }
    }
}