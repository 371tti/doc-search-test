use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicU64, Ordering};

use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use rayon::prelude::*;
use tf_idf_vectorizer::{Corpus, Hits, SimilarityAlgorithm, TFIDFData, TFIDFVectorizer, TokenFrequency};
use walkdir::WalkDir;

use crate::tokenize::SudachiTokenizer;

pub mod tokenize;

type DynError = Box<dyn std::error::Error + Send + Sync>;

#[derive(Parser)]
#[command(name = "doc-search-test", version, about = "Build/load a TF-IDF index and run searches")]
struct Cli {
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
}

fn main() -> Result<(), DynError> {
    let cli = Cli::parse();

    match cli.command {
        Command::Index { docs, index } => {
            build_index(&docs, &index)?;
        }
        Command::Search { index, query, top } => {
            run_search(&index, &query, top)?;
        }
    }

    Ok(())
}

fn build_index(docs_path: &Path, index_dir: &Path) -> Result<(), DynError> {
    if !docs_path.exists() {
        return Err(format!("Docs path not found: {}", docs_path.display()).into());
    }
    std::fs::create_dir_all(index_dir)?;

    let files = collect_doc_files(docs_path)?;
    if files.is_empty() {
        return Err(format!("No files found under: {}", docs_path.display()).into());
    }

    let mut indexer = DocIndexer::new();
    let tokenizer = indexer.tokenizer.clone();

    eprintln!("Indexing {} files...", files.len());
    let started = Instant::now();

    let pb = ProgressBar::new(files.len() as u64);
    pb.set_draw_target(ProgressDrawTarget::stderr_with_hz(10));
    pb.enable_steady_tick(Duration::from_millis(120));
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({percent}%) {per_sec} ETA {eta_precise}",
        )
        .unwrap()
        .progress_chars("=>-"),
    );

    // Parallel read+tokenize, then add to the vectorizer sequentially.
    // We increment progress for every attempted file. Non-text / non-UTF8 are skipped
    // but only summarized at the end to avoid flooding the console.
    let skipped_non_utf8 = AtomicU64::new(0);
    let skipped_io = AtomicU64::new(0);

    let mut items: Vec<(String, TokenFrequency, u64)> = files
        .par_iter()
        .map(|(key, path)| {
            let res: Result<(String, TokenFrequency, u64), ()> = (|| {
                let text = match std::fs::read_to_string(path) {
                    Ok(t) => t,
                    Err(e) if e.kind() == std::io::ErrorKind::InvalidData => {
                        skipped_non_utf8.fetch_add(1, Ordering::Relaxed);
                        return Err(());
                    }
                    Err(_) => {
                        skipped_io.fetch_add(1, Ordering::Relaxed);
                        return Err(());
                    }
                };
                let (tokens, token_sum) = tokenizer
                    .mix_doc_tokenizer(&text)
                    .map_err(|_| ())?;
                let mut freq = TokenFrequency::new();
                freq.add_tokens(&tokens);
                Ok((key.clone(), freq, token_sum))
            })();
            pb.inc(1);
            res
        })
        .filter_map(|res| res.ok())
        .collect();

    pb.finish_and_clear();

    // Keep deterministic build order (useful for debugging/reproducibility).
    items.sort_by(|a, b| a.0.cmp(&b.0));

    let mut total_tokens: u64 = 0;
    for (key, freq, token_sum) in items {
        indexer.vectorizer.add_doc(key, &freq);
        total_tokens = total_tokens.saturating_add(token_sum);
    }

    indexer.save_to(index_dir)?;

    let elapsed = started.elapsed().as_secs_f64().max(0.000_001);
    let docs_indexed = indexer.corpus.get_doc_num() as f64;
    let docs_per_sec = docs_indexed / elapsed;
    let tokens_per_sec = (total_tokens as f64) / elapsed;
    eprintln!(
        "Done. docs={} tokens={} skipped_non_utf8={} skipped_io={} elapsed={:.2}s docs/s={:.2} tokens/s={:.2}",
        docs_indexed as u64,
        total_tokens,
        skipped_non_utf8.load(Ordering::Relaxed),
        skipped_io.load(Ordering::Relaxed),
        elapsed,
        docs_per_sec,
        tokens_per_sec
    );
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

fn collect_doc_files(root: &Path) -> Result<Vec<(String, PathBuf)>, DynError> {
    let meta = std::fs::metadata(root)?;
    if meta.is_file() {
        let key = root
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| root.to_string_lossy().to_string());
        return Ok(vec![(key, root.to_path_buf())]);
    }

    let mut out = Vec::new();

    let walker = WalkDir::new(root).follow_links(false).into_iter().filter_entry(|e| {
        // Avoid indexing build artifacts / repo metadata by default.
        // (This mainly matters when users pass something like `--docs ./`.)
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
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path().to_path_buf();
        let rel = match path.strip_prefix(root) {
            Ok(p) => p,
            Err(_) => path.as_path(),
        };
        let key = rel.to_string_lossy().replace('\\', "/");
        out.push((key, path));
    }

    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

pub struct DocIndexer {
    pub corpus: Arc<Corpus>,
    pub vectorizer: TFIDFVectorizer<u16>,
    pub tokenizer: SudachiTokenizer,
}

impl DocIndexer {
    pub fn new() -> Self {
        let corpus = Arc::new(Corpus::new());
        let vectorizer = TFIDFVectorizer::new(corpus.clone());
        let tokenizer = SudachiTokenizer::new().unwrap();
        DocIndexer { corpus, vectorizer, tokenizer }
    }

    pub fn save_to(&self, index_dir: &Path) -> Result<(), DynError> {
        std::fs::create_dir_all(index_dir)?;
        let corpus = std::fs::File::create(index_dir.join("corpus.cbor"))?;
        let vector = std::fs::File::create(index_dir.join("vector.cbor"))?;
        ciborium::into_writer(&*self.corpus, corpus)?;
        ciborium::into_writer(&self.vectorizer, vector)?;
        Ok(())
    }

    pub fn load_from(index_dir: &Path) -> Result<Self, DynError> {
        let corpus_file = std::fs::File::open(index_dir.join("corpus.cbor"))?;
        let vector_file = std::fs::File::open(index_dir.join("vector.cbor"))?;
        let corpus: Corpus = ciborium::from_reader(corpus_file)?;
        let corpus = Arc::new(corpus);
        let vectorizer: TFIDFData<u16> = ciborium::from_reader(vector_file)?;
        let vectorizer = vectorizer.into_tf_idf_vectorizer(corpus.clone());
        let tokenizer = SudachiTokenizer::new()?;
        Ok(DocIndexer { corpus, vectorizer, tokenizer })
    }

    pub fn query_to_freq(&self, query: &str) -> Result<TokenFrequency, DynError> {
        let tokenized = self.tokenizer.mix_query_tokenizer(query)?;
        let mut freq = TokenFrequency::new();
        freq.add_tokens(&tokenized);
        Ok(freq)
    }

    pub fn doc_to_freq(&self, doc: &str) -> Result<(TokenFrequency, u64), DynError> {
        let (tokenized, token_sum) = self.tokenizer.mix_doc_tokenizer(doc)?;
        let mut freq = TokenFrequency::new();
        freq.add_tokens(&tokenized);
        Ok((freq, token_sum))
    }

    pub fn add_document(&mut self, doc: &str, key: &str) -> Result<u64, DynError> {
        let (freq, token_sum) = self.doc_to_freq(doc)?;
        self.vectorizer.add_doc(key.to_string(), &freq);
        Ok(token_sum)
    }

    pub fn search_doc(&mut self, query: &str) -> Result<Hits<String>, DynError> {
        let query_freq = self.query_to_freq(query)?;
        let mut results = self.vectorizer.similarity(&query_freq, &SimilarityAlgorithm::CosineSimilarity);
        results.sort_by_score();
        Ok(results)
    }
}