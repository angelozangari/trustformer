struct Transformer;

impl Transformer {
    fn new() -> Self {
        Transformer
    }

    fn train(&self) {
        // add training logic
    }

    fn inference(&self, input: &str) -> String {
        // do inference
        "out".to_string()
    }
}

fn main() {
    let input = "how many r's in strawberry";

    let model = Transformer::new();

    model.train();

    let output = model.inference(input);

    println!("model response: {output}")
}
