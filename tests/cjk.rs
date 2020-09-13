#[cfg(feature = "levenshtein")]
use fst::automaton::Levenshtein;
#[cfg(feature = "levenshtein")]
use fst::{self, IntoStreamer, Set};

// To run and only run this test case:
//  `cargo test --all-features --test cjk`

// Returns elements in `set` which fuzzy-match `key` by distance 1.
//
// A levenshtein automaton will be constructed for the `key`, and a graph
// will be generated for the automaton in DOT language at `output_dot_path`.
#[cfg(feature = "levenshtein")]
fn search<T: std::convert::AsRef<[u8]>>(
    set: &Set<T>,
    key: &str,
    output_dot_path: &str,
) -> Vec<String> {
    use std::fs::File;
    use std::io::BufWriter;

    let lev = Levenshtein::new(key, 1).unwrap();

    // Generate a graph for the automaton.
    let file = File::create(output_dot_path).expect("create dot file error");
    let mut writer = BufWriter::new(file);
    lev.to_graphviz("dfa", &mut writer).expect("to_graphviz() error");

    // Apply our fuzzy query to the set we built.
    let stream = set.search(lev).into_stream();
    stream.into_strs().unwrap()
}

#[cfg(feature = "levenshtein")]
#[test]
fn levenshtein_cjk() {
    // A convenient way to create sets in memory.
    let keys = vec!["北", "北半球", "北方"];
    let set = Set::from_iter(keys).unwrap();

    assert_eq!(search(&set, "北a", "dfa.dot"), vec!["北", "北方"]);
    // NOTE: command to convert the dot file to image file: `dot -Tpng < dfa.dot > dfa.png`

    // The following case will fail, demonstrating that the current levenshtein impl has
    // bugs in multi-byte characters processing.
    assert_eq!(search(&set, "北境", "bad.dot"), vec!["北", "北方"]);
}
