use ndarray::Array2;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_shapes() {
        let a = Tensor::from_vec(2, 3, vec![1.,2.,3., 4.,5.,6.]);
        let b = Tensor::from_vec(3, 2, vec![7.,8., 9.,10., 11.,12.]);
        let c = a.matmul(&b);
        assert_eq!(c.0.dim(), (2,2))
    }

    #[test]
    fn transpose_shapes() {
        let a = Tensor::zeros(2, 3);
        let at = a.transpose();
        assert_eq!(at.shape(), (3,2));
    }

    #[test]
    fn linear_forward_shape() {
        let lin = Linear::new(3, 4);
        let x = Tensor::zeros(2, 3);
        let y = lin.forward(&x);
        assert_eq!(y.shape(), (2,4));
    }

    #[test]
    fn block_forward_shape() {
        let d = 5; 
        let blk = Block::new(d);
        let x = Tensor::zeros(7, 5); // 7 "tokens" flattened, d features
        let y = blk.forward(&x);
        assert_eq!(y.shape(), (7, d));
    }
}

#[derive(Clone, Debug)]
struct Block {
    proj: Linear, // [d_model, d_model]
}

impl Block {
    fn new(d_model: usize) -> Self {
        Block { proj: Linear::new(d_model, d_model) }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        // residual: x + proj(x)
        x.add(&self.proj.forward(x))
    }
}

#[derive(Clone, Debug)]
struct Linear {
    w: Tensor, // [in, out]
}

impl Linear {
    fn new(in_dim: usize, out_dim: usize) -> Self {
        // deterministic init for now (all zeros)
        Linear { w: Tensor::zeros(in_dim, out_dim) }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        x.matmul(&self.w)
    }
}

#[derive(Clone, Debug)]
struct Tensor(Array2<f32>);

impl Tensor {
    fn zeros(rows: usize, cols: usize) -> Self {
        Tensor(Array2::zeros((rows, cols)))
    }
 
    fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        Tensor(Array2::from_shape_vec((rows, cols), data).expect("bad shape"))
    }

    fn matmul(&self, other: &Tensor) -> Tensor {
        let (a_r, a_c) = self.0.dim();
        let (b_r, b_c) = other.0.dim();
        assert!(a_c == b_r, "matmul shape mismatch: ({a_r},{a_c})x({b_r},{b_c})");
        Tensor(self.0.dot(&other.0))
    }
    fn add(&self, other: &Tensor) -> Tensor {
        let a = self.0.dim();
        let b = other.0.dim();
        assert!(a == b, "add shape mismatch: {:?} vs {:?}", a, b);
        Tensor(&self.0 + &other.0)
    }

    fn shape(&self) -> (usize, usize) {
        self.0.dim()
    }

    fn transpose(&self) -> Tensor {
        Tensor(self.0.t().to_owned())
    }

}

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
