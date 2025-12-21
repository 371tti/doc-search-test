use std::collections::HashSet;
use std::io::{self, BufRead, BufReader, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use clap::{Parser, Subcommand};
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use rayon::{ThreadPoolBuilder, prelude::*};
use tf_idf_vectorizer::TokenFrequency;
use walkdir::WalkDir;

use crate::indexer::{DocIndexer, DynError};
use crate::tokenize::SudachiTokenizer;

#[derive(Parser)]
#[command(name = "doc-search-test", version, about = "Build/load a TF-IDF index and run searches")]
pub struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Build an index from documents under the given path
    Index {
        /// Document directory (recursively scanned) or a single file
        #[arg(long)]
        docs: PathBuf,

        /// Output directory for the index
        #[arg(long, default_value = "./index")]
        index: PathBuf,
    },

    /// Search using an existing index
    Search {
        /// Index directory
        #[arg(long, default_value = "./index")]
        index: PathBuf,

        /// Query string
        #[arg(long)]
        query: String,

        /// Max results to print
        #[arg(long, default_value_t = 20)]
        top: usize,
    },

    /// Load an index once and accept multiple queries from stdin
    Shell {
        /// Index directory
        #[arg(long, default_value = "./index")]
        index: PathBuf,

        /// Max results to print
        #[arg(long, default_value_t = 20)]
        top: usize,
    },
}

pub fn run() -> Result<(), DynError> {
    let cli = Cli::parse();

    match cli.command {
        Command::Index { docs, index } => build_index(&docs, &index),
        Command::Search { index, query, top } => run_search(&index, &query, top),
        Command::Shell { index, top } => run_shell(&index, top),
    }
}

fn build_index(docs_path: &Path, index_dir: &Path) -> Result<(), DynError> {
    if !docs_path.exists() {
        return Err(format!("Docs path not found: {}", docs_path.display()).into());
    }
    std::fs::create_dir_all(index_dir)?;

    // Count files first to provide an accurate ETA without keeping all paths in memory.
    let total_files: u64 = {
        let meta = std::fs::metadata(docs_path)?;
        if meta.is_file() {
            1
        } else {
            let mut count: u64 = 0;
            let counter = ProgressBar::new_spinner();
            counter.set_draw_target(ProgressDrawTarget::stderr_with_hz(10));
            counter.enable_steady_tick(Duration::from_millis(120));
            counter.set_style(
                ProgressStyle::with_template("{spinner:.green} counting files... {pos}")
                    .unwrap()
                    .progress_chars("=>-"),
            );

            let walker = WalkDir::new(docs_path).follow_links(false).into_iter().filter_entry(|e| {
                let Some(name) = e.file_name().to_str() else {
                    return true;
                };
                !matches!(name, "target" | ".git" | "index")
            });

            for entry in walker {
                let entry = match entry {
                    Ok(e) => e,
                    Err(_) => continue,
                };
                if entry.file_type().is_file() {
                    count += 1;
                    if count % 2048 == 0 {
                        counter.set_position(count);
                    }
                }
            }

            counter.finish_and_clear();
            count
        }
    };

    if total_files == 0 {
        return Err(format!("No files found under: {}", docs_path.display()).into());
    }

    let mut indexer = DocIndexer::new();
    let tokenizer = indexer.tokenizer.clone();

    eprintln!("Indexing {} files...", total_files);
    let started = Instant::now();

    

    let m = MultiProgress::new();
    m.set_draw_target(ProgressDrawTarget::stderr_with_hz(10));

    // total style
    let sty_total = ProgressStyle::with_template(
        "{spinner:.green} [{elapsed_precise}] {wide_bar:.cyan/blue} {pos}/{len} ({percent}%) {per_sec} {msg} ETA {eta_precise}",
    )
    .unwrap()
    .progress_chars("##-");

    // worker style（prefix に worker id を入れる）
    let sty_worker = ProgressStyle::with_template(
        "    worker-{prefix:<2} {spinner:.cyan} {msg}",
    )
    .unwrap();

    // total PB（1本だけ）
    let pb_total = m.add(ProgressBar::new(total_files));
    pb_total.set_style(sty_total);
    pb_total.enable_steady_tick(Duration::from_millis(120));

    // worker PBs（固定本数：rayon threads）
    ThreadPoolBuilder::new()
        .num_threads(8)
        .build_global()
        .expect("rayon global pool already initialized");
    let n_workers = rayon::current_num_threads().max(1);
    let worker_pbs: Vec<ProgressBar> = (0..n_workers)
        .map(|i| {
            let pb = m.add(ProgressBar::new_spinner());
            pb.set_style(sty_worker.clone());
            pb.set_prefix(format!("{i}"));
            pb.set_message("idle");
            pb.enable_steady_tick(Duration::from_millis(80));
            pb
        })
        .collect();

    // counters
    let skipped_non_utf8 = AtomicU64::new(0);
    let skipped_io = AtomicU64::new(0);
    let skipped_tokenize = AtomicU64::new(0);
    let processed_tokens = AtomicU64::new(0);

    let mut total_tokens: u64 = 0;

    // Process in batches so we don't keep all (path -> TokenFrequency) in memory.
    const BATCH_SIZE: usize = 512;

    let mut handle_batch = |batch: &[(String, PathBuf)]| {
        // NOTE: batch PB は作らない（表示しない）

        let results: Vec<(String, TokenFrequency, u64)> = batch
            .par_iter()
            .filter_map(|(key, path)| {
                // どの worker 行を更新するか：rayon のスレッド index を使う（Atomic不要で速い）
                let tid = rayon::current_thread_index().unwrap_or(0);
                let wpb = &worker_pbs[tid.min(worker_pbs.len() - 1)];

                // “今なにしてるか”
                wpb.set_message(format!("reading {key}"));
                let res: Result<(String, TokenFrequency, u64), ()> = (|| {
                    wpb.set_message(format!("tokenize {key}"));

                    let (freq, token_sum) = match build_doc_freq_streaming(&tokenizer, path) {
                        Ok(v) => v,
                        Err(e) if e.kind() == ErrorKind::InvalidData => {
                            skipped_non_utf8.fetch_add(1, Ordering::Relaxed);
                            wpb.set_message(format!("skip (utf8) {key}"));
                            return Err(());
                        }
                        Err(e) if e.kind() == ErrorKind::Other => {
                            skipped_tokenize.fetch_add(1, Ordering::Relaxed);
                            wpb.set_message(format!("skip (tokenize) {key}"));
                            return Err(());
                        }
                        Err(_) => {
                            skipped_io.fetch_add(1, Ordering::Relaxed);
                            wpb.set_message(format!("skip (io) {key}"));
                            return Err(());
                        }
                    };

                    processed_tokens.fetch_add(token_sum, Ordering::Relaxed);
                    Ok((key.clone(), freq, token_sum))
                })();

                pb_total.inc(1);

                res.ok()
            })
            .collect();

        for (key, freq, token_sum) in results {
            indexer.vectorizer.add_doc(key, &freq);
            total_tokens = total_tokens.saturating_add(token_sum);
        }

        // tokens/s はたまに更新（軽く）
        let pos = pb_total.position();
        if pos % 2048 == 0 {
            let elapsed = started.elapsed().as_secs_f64().max(0.000_001);
            let toks = processed_tokens.load(Ordering::Relaxed) as f64;
            pb_total.set_message(format!("tokens/s={:.2}", toks / elapsed));
        }

        // worker 表示を idle に戻したいならここ（好み）
        for pb in &worker_pbs {
            pb.set_message("idle");
        }
    };


    let meta = std::fs::metadata(docs_path)?;
    if meta.is_file() {
        let key = docs_path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| docs_path.to_string_lossy().to_string());
        handle_batch(&[(key, docs_path.to_path_buf())]);
    } else {
        let walker = WalkDir::new(docs_path).follow_links(false).into_iter().filter_entry(|e| {
            let Some(name) = e.file_name().to_str() else {
                return true;
            };
            !matches!(name, "target" | ".git" | "index")
        });

        let mut batch: Vec<(String, PathBuf)> = Vec::with_capacity(BATCH_SIZE);
        for entry in walker {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };
            if !entry.file_type().is_file() {
                continue;
            }
            let path = entry.path().to_path_buf();
            let rel = match path.strip_prefix(docs_path) {
                Ok(p) => p,
                Err(_) => path.as_path(),
            };
            let key = rel.to_string_lossy().replace('\\', "/");
            batch.push((key, path));
            if batch.len() >= BATCH_SIZE {
                handle_batch(&batch);
                batch.clear();
            }
        }
        if !batch.is_empty() {
            handle_batch(&batch);
        }
    }

    indexer.save_to(index_dir)?;

    let elapsed = started.elapsed().as_secs_f64().max(0.000_001);
    let docs_indexed = indexer.corpus.get_doc_num() as f64;
    let docs_per_sec = docs_indexed / elapsed;
    let tokens_per_sec = (total_tokens as f64) / elapsed;
    eprintln!(
        "Done. docs={} tokens={} skipped_non_utf8={} skipped_io={} skipped_tokenize={} elapsed={:.2}s docs/s={:.2} tokens/s={:.2}",
        docs_indexed as u64,
        total_tokens,
        skipped_non_utf8.load(Ordering::Relaxed),
        skipped_io.load(Ordering::Relaxed),
        skipped_tokenize.load(Ordering::Relaxed),
        elapsed,
        docs_per_sec,
        tokens_per_sec
    );

    Ok(())
}

fn build_doc_freq_streaming(tokenizer: &SudachiTokenizer, path: &Path) -> Result<(TokenFrequency, u64), std::io::Error> {
    const FLUSH_THRESHOLD_BYTES: usize = 256 * 1024;
    const DELIMS: &str = "。．、？！.!?；;,，\n\r";

    let mut c_membership: HashSet<Box<str>> = HashSet::new();
    let mut freq = TokenFrequency::new();
    let mut token_sum: u64 = 0;

    // Pass 1: Mode::C tokens
    {
        let file = std::fs::File::open(path)?;
        let mut reader = BufReader::new(file);
        for_each_text_chunk(&mut reader, FLUSH_THRESHOLD_BYTES, DELIMS, |chunk| {
            let c = tokenizer
                .tokenize(chunk, sudachi::prelude::Mode::C)
                .map_err(|e| std::io::Error::new(ErrorKind::Other, e.to_string()))?;
            let c_tokens = c.tokens();
            token_sum = token_sum.saturating_add(c_tokens.len() as u64);
            for t in &c_tokens {
                c_membership.insert(t.clone());
            }
            freq.add_tokens(&c_tokens);
            Ok(())
        })?;
    }

    // Pass 2: Mode::A tokens
    {
        let file = std::fs::File::open(path)?;
        let mut reader = BufReader::new(file);
        for_each_text_chunk(&mut reader, FLUSH_THRESHOLD_BYTES, DELIMS, |chunk| {
            let a = tokenizer
                .tokenize(chunk, sudachi::prelude::Mode::A)
                .map_err(|e| std::io::Error::new(ErrorKind::Other, e.to_string()))?;
            let a_tokens = a.tokens();
            for t in a_tokens {
                if !c_membership.contains(&t) {
                    freq.add_token(&t);
                }
            }
            let speech_tokens = a.speech_tokens();
            freq.add_tokens(&speech_tokens);
            Ok(())
        })?;
    }

    Ok((freq, token_sum))
}

fn for_each_text_chunk<R: BufRead>(
    reader: &mut R,
    flush_threshold_bytes: usize,
    delimiters: &str,
    mut f: impl FnMut(&str) -> Result<(), std::io::Error>,
) -> Result<(), std::io::Error> {
    let mut pending = String::new();
    let mut line = Vec::new();

    loop {
        line.clear();
        let n = reader.read_until(b'\n', &mut line)?;
        if n == 0 {
            break;
        }

        let s = std::str::from_utf8(&line)
            .map_err(|_| std::io::Error::new(ErrorKind::InvalidData, "stream did not contain valid UTF-8"))?;
        pending.push_str(s);

        if pending.len() >= flush_threshold_bytes {
            flush_pending(&mut pending, delimiters, &mut f)?;
        }
    }

    if !pending.is_empty() {
        flush_pending(&mut pending, delimiters, &mut f)?;
    }

    Ok(())
}

fn flush_pending(
    pending: &mut String,
    delimiters: &str,
    f: &mut impl FnMut(&str) -> Result<(), std::io::Error>,
) -> Result<(), std::io::Error> {
    while pending.len() >= 64 * 1024 {
        let cut = pending
            .char_indices()
            .rev()
            .find(|&(_, ch)| delimiters.contains(ch))
            .map(|(idx, ch)| idx + ch.len_utf8())
            .unwrap_or(64 * 1024);

        let chunk: String = pending.drain(..cut).collect();
        if !chunk.trim().is_empty() {
            f(&chunk)?;
        }
    }
    Ok(())
}

fn run_search(index_dir: &Path, query: &str, top: usize) -> Result<(), DynError> {
    let mut indexer = DocIndexer::load_from(index_dir)?;
    let results = indexer.search_doc(query)?;

    for (rank, (key, score, doc_len)) in results.list.into_iter().take(top).enumerate() {
        println!("{}\t{:.6}\t{}\t{}", rank + 1, score, doc_len, key);
    }

    Ok(())
}

fn run_shell(index_dir: &Path, top: usize) -> Result<(), DynError> {
    let mut indexer = DocIndexer::load_from(index_dir)?;

    println!("Shell mode. Type a query and press Enter.");
    eprintln!("Type 'exit' or 'quit' to stop.");

    let stdin = io::stdin();
    let mut line = String::new();
    loop {
        eprint!("> ");
        io::stderr().flush().ok();

        line.clear();
        let n = stdin.read_line(&mut line)?;
        if n == 0 {
            break;
        }
        let q = line.trim();
        if q.is_empty() {
            continue;
        }
        if matches!(q, "exit" | "quit") {
            break;
        }

        
        let results = match indexer.search_doc(q) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[error] {e}");
                continue;
            }
        };
        

        for (rank, (key, score, doc_len)) in results.list.into_iter().take(top).enumerate() {
            println!("{}\t{:.6}\t{}\t{}", rank + 1, score, doc_len, key);
        }
    }

    Ok(())
}
