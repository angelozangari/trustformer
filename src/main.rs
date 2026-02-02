use ndarray::{Array1, Array2, Axis};

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

    #[test]
    fn layernorm_shape() {
        let x = Tensor::from_vec(2, 3, vec![1.,2.,3., 4., 5.,6.]);
        let ln = LayerNorm::new(3, 1e-5);
        let y = ln::forward(&x);
        assert_eq!(y.shape(), (2,3));
    }

    #[test]
    fn layernorm_row_stats_approx() {
        let x = Tensor::from_vec(1, 4, vec![1., 2., 3., 4.]); // non-constant row
        let ln = LayerNorm::new(4, 1e-5);
        let y = ln.forward(&x);

        let row = y.0.row(0);
        let mean = row.sum() / 4.0;
        let mut var = 0.0;
        for v in row.iter() {
            var += (*v - mean) * (*v - mean);
        }
        var /= 4.0;

        assert!(mean.abs() < 1e-4, "mean={mean}");
        assert!((var - 1.0).abs() < 1e-3, "var={var}");
    }

    #[test]
    fn self_attention_shape() {
        let d = 8;
        let n = 5;
        let attn = SelfAttention::new(d);
        let x = Tensor::zeros(n, d);
        let y = attn.forward(&x);
        assert_eq!(y.shape(), (n, d));
    }
}

#[derive(Clone, Debug)]
struct Block {
    ln: LayerNorm,
    attn: SelfAttention,
}

impl Block {
    fn new(d_model: usize) -> Self {
        Block { 
            ln: LayerNorm::new(d_model, 1e-5),
            attn: SelfAttention::new(d_model),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let h = self.ln.forward(x);
        // residual: x + attn(h)
        x.add(&self.attn.forward(&h))
    }
}

#[derive(Clone, Debug)]
struct LayerNorm {
    gamma: Array1<f32>, // [d]
    beta: Array1<f32>,  // [d]
    eps: f32,
}

impl LayerNorm {
    fn new(d_model: usize, eps: f32) -> Self {
        LayerNorm {
            gamma: Array1::ones(d_model),
            beta: Array1::zeros(d_model),
            eps,
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let (n, d) = x.shape();
        assert_eq!(self.gamma.len(), d, "LayerNorm gamma shape mismatch");
        assert_eq!(self.beta.len(), d, "LayerNorm beta shape mismatch");

        let mut y = x.0.clone();

        for mut row in y.axis_iter_mut(Axis(0)) {
            let mean = row.sum() / (d as f32);
            let mut var = 0.0f32;
            for v in row.iter() {
                let t = *v - mean;
                var += t * t;
            }
            var /= f as f32;
            let inv_std = 1.0f32 / (var + self.eps).sqrt();

            for j in 0..d {
                let norm = (row[j] - mean) * inv_std;
                row[j] = norm * self.gamma[j] + self.beta[j];
            }
        }
        
        Tensor(y)
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
struct SelfAttention {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    d_model: usize,
}

impl SelfAttention {
    fn new(d_model: usize) -> Self {
        SelfAttention { 
            wq: Linear::new(d_model, d_model),
            wk: Linear::new(d_model, d_model),
            wv: Linear::new(d_model, d_model),
            wo: Linear::new(d_model, d_model),
            d_model,
        }
    }
    
    fn forward(&self, x: &Tensor) -> Tensor {
        // x: [n, d]
        let q = self.wq.forward(x); // [n, d]
        let k = self.wk.forward(x); // [n, d]
        let v = self.wv.forward(x); // [n, d]

        // scores: [n, n] = Q K^T / sqrt(d)
        let scale = 1.0f32 / (self.d_model as f32).sqrt();
        let scores = q.matmul(&k.transpose()).scale(scale); // [n, n]

        // note: no softmax yet; this is intentionally wrong numerically but shape-correct
        let ctx = scores.matmul(&v); // [n, d]
        self.wo.forward(&ctx) // [n, d]
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

    fn scale(&self, s: f32) -> Tensor {
        Tensor(self.0.mapv(|x| x*s))
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
