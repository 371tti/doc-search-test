use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use half::f16;
use indicatif::{HumanBytes, ProgressBar, ProgressDrawTarget, ProgressStyle};
use jwalk::WalkDir;
use tf_idf_vectorizer::{Corpus, Hits, SimilarityAlgorithm, TFIDFData, TFIDFVectorizer, TokenFrequency};

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
    pub tokenize_sender: flume::Sender<WithKey<String>>,
    pub tokenize_receiver: flume::Receiver<WithKey<String>>,
    pub vectorize_sender: flume::Sender<WithKey<(Vec<Box<str>>, u64)>>,
    pub vectorize_receiver: flume::Receiver<WithKey<(Vec<Box<str>>, u64)>>,
}

#[derive(Default)]
pub struct PipelineStats {
    // path stage
    pub paths_sent: AtomicU64,

    // read stage
    pub read_ok: AtomicU64,
    pub read_err: AtomicU64,
    pub utf8_err: AtomicU64,
    pub bytes_read: AtomicU64,

    // tokenize stage
    pub tok_ok: AtomicU64,
    pub tok_err: AtomicU64,

    // index stage
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
                let key = path.to_string_lossy().into_owned();

                let bytes = match std::fs::read(&path) {
                    Ok(b) => b,
                    Err(_) => { stats.read_err.fetch_add(1, Ordering::Relaxed); continue; }
                };
                stats.bytes_read.fetch_add(bytes.len() as u64, Ordering::Relaxed);

                let content = match std::str::from_utf8(&bytes) {
                    Ok(s) => s,
                    Err(_) => { stats.utf8_err.fetch_add(1, Ordering::Relaxed); continue; }
                };

                if sender.send(WithKey { key, value: content.to_owned() }).is_err() {
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
                let tokenized = match tokenizer.mix_doc_tokenizer(&text.value) {
                    Ok(t) => t,
                    Err(_) => { stats.tok_err.fetch_add(1, Ordering::Relaxed); continue; }
                };
                if sender.send(WithKey { key: text.key, value: tokenized }).is_err() {
                    return;
                }
                stats.tok_ok.fetch_add(1, Ordering::Relaxed);
            }
        })
    }
    
    fn spawn_metrics_thread(stats: Arc<PipelineStats>, total_files: u64) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || {
            let pb = ProgressBar::new(total_files);
            pb.set_draw_target(ProgressDrawTarget::stderr_with_hz(8));
            pb.set_style(
                ProgressStyle::with_template(
                    "{spinner:.green} [{elapsed_precise}] {wide_bar:.cyan/blue} {pos}/{len} ({percent}%) \
                    idx {per_sec} ETA {eta_precise}\n{msg}"
                )
                .unwrap()
                .progress_chars("#>-")
                .tick_strings(&[
                    "|", "/", "-", "\\",
                ]),
            );
            pb.enable_steady_tick(Duration::from_millis(120));

            let start = Instant::now();
            let mut last = Instant::now();
            let mut last_read_ok = 0u64;
            let mut last_tok_ok  = 0u64;
            let mut last_added   = 0u64;
            let mut last_bytes   = 0u64;

            loop {
                std::thread::sleep(Duration::from_millis(500));

                let read_ok = stats.read_ok.load(Ordering::Relaxed);
                let tok_ok  = stats.tok_ok.load(Ordering::Relaxed);
                let added   = stats.docs_added.load(Ordering::Relaxed);
                let bytes   = stats.bytes_read.load(Ordering::Relaxed);

                let read_err = stats.read_err.load(Ordering::Relaxed);
                let utf8_err = stats.utf8_err.load(Ordering::Relaxed);
                let tok_err  = stats.tok_err.load(Ordering::Relaxed);

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

                pb.set_message(format!(
                    "read:  {:>8.1} files/s  ({read_ok} ok, {read_err} err, {utf8_err} utf8)\n\
                    tok:   {:>8.1} docs/s   ({tok_ok} ok, {tok_err} err)\n\
                    index: {:>8.1} docs/s   ({added} added)\n\
                    io:    {:>8.1} MiB/s    (total {:.1} GiB)\n\
                    uptime: {:?}",
                    read_rps,
                    tok_rps,
                    add_rps,
                    mbps,
                    bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                    start.elapsed(),
                ));

                // すべて追加し終わったら終了
                if added >= total_files {
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
        let file_reader_threads = 8; // 必要に応じて数を調整
        let mut file_reader_handles = Vec::with_capacity(file_reader_threads);
        for _ in 0..file_reader_threads {
            file_reader_handles.push(DocIndexer::spawm_file_reader_thread(&queues, stats.clone()));
        }

        // トークナイズスレッドを複数立てる
        let tokenize_threads = 16; // 必要に応じて数を調整
        let mut tokenize_handles = Vec::with_capacity(tokenize_threads);
        for _ in 0..tokenize_threads {
            tokenize_handles.push(self.spawn_tokenize_thread(&queues, stats.clone()));
        }

        drop(queues.file_reader_sender);
        drop(queues.tokenize_sender);
        drop(queues.vectorize_sender);

        for WithKey { key, value: (tokens, _token_num) } in queues.vectorize_receiver.iter() {
            let doc = TokenFrequency::from(tokens.as_slice());
            self.vectorizer.add_doc(key, &doc);
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

    pub fn query_to_freq(&self, query: &str) -> Result<TokenFrequency, DynError> {
        let tokenized  = self.tokenizer.pure_doc_tokenizer(query)?;
        println!("Query tokens: {:?}", tokenized);
        let mut freq = TokenFrequency::new();
        freq.add_tokens(&tokenized.0);
        Ok(freq)
    }

    pub fn search_doc(&mut self, query: &str, algorithm: SimilarityAlgorithm) -> Result<Hits<String>, DynError> {
        let inst = std::time::Instant::now();
        let query_freq = self.query_to_freq(query)?;
        let mut results = self.vectorizer.similarity(&query_freq, &algorithm);
        results.sort_by_score_desc();
        let elapsed = inst.elapsed().as_millis();
        println!("Found {} results in {} ms.", results.list.len(), elapsed);
        Ok(results)
    }
}
