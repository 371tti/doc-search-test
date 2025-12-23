use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::{Parser, Subcommand};
use tf_idf_vectorizer::SimilarityAlgorithm;

use crate::indexer::{DocIndexer, DynError};


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

pub fn build_index(docs_path: &Path, index_dir: &Path) -> Result<(), DynError> {
    if !docs_path.exists() {
        return Err(format!("Docs path not found: {}", docs_path.display()).into());
    }
    std::fs::create_dir_all(index_dir)?;

    let started = Instant::now();
    let mut indexer = DocIndexer::new();

    indexer.build_index(docs_path)?;

    /* ---------- Save ---------- */
    indexer.save_to(index_dir)?;

    let elapsed = started.elapsed().as_secs_f64().max(1e-6);
    let docs = indexer.corpus.get_doc_num() as f64;

    println!(
        "Indexing completed: {} docs in {:.2} sec ({:.2} docs/sec)",
        docs as u64,
        elapsed,
        docs / elapsed
    );

    Ok(())
}

fn run_search(index_dir: &Path, query: &str, top: usize) -> Result<(), DynError> {
    let mut indexer = DocIndexer::load_from(index_dir)?;
    let mut results = indexer.search_doc(query, SimilarityAlgorithm::Contains)?;

    results.sort_by_score_desc();
    results.top_k(top);
    println!("results: {}", results);

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

        // 先頭にアルゴリズム名があればそのアルゴリズムを適用する
        // 例: "bm25: 検索クエリ" のように入力
        let (algorithm, q) = if let Some(idx) = q.find(':') {
            let (alg, rest) = q.split_at(idx);
            let alg = alg.trim().to_lowercase();
            let rest = rest[1..].trim(); // ':' の次から
            match alg.as_str() {
                "bm25" => (SimilarityAlgorithm::BM25(1.2, 0.75), rest),
                "dot" => {
                    (SimilarityAlgorithm::Dot, rest)
                }
                "contains" => {
                    (SimilarityAlgorithm::Contains, rest)
                }
                "cosine" => {
                    (SimilarityAlgorithm::CosineSimilarity, rest)
                }
                _ => {
                    (SimilarityAlgorithm::CosineSimilarity, q)
                }
            }
        } else {
            (SimilarityAlgorithm::CosineSimilarity, q)
        };
        
        let mut results = match indexer.search_doc(q, algorithm) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[error] {e}");
                continue;
            }
        };
        
        results.top_k(top);
        println!("results: {}", results);
    }

    Ok(())
}
