

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

pub struct Network {
    layers: Vec<Layer>
}

struct Layer {
    neurons: Vec<Neuron>
}
struct Neuron {
    bias: f32,
    weights: Vec<f32>
}

pub struct LayerTopology {
    pub neurons: usize,
}

impl Network {

    pub fn random(layers: &[LayerTopology]) -> Self {
        assert!(layers.len() > 1);
        
        let layers = layers
                        .windows(2)
                        .map(|layers| {
                            Layer::random(layers[0].neurons, layers[1].neurons)
                        })
                        .collect();
        Self { layers }
    }

    pub fn propagate(&self, mut inputs: Vec<f32>) -> Vec<f32> {
        for layer in &self.layers {
            inputs = layer.propagate(inputs);
        }
        inputs
    }
}

impl Layer {
    fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.neurons
            .iter()
            .map(|neurou| neurou.propagate(&inputs))
            .collect()
    }
    pub fn random(input_neurons: usize, output_neurons: usize) -> Self {
        let mut neurons = Vec::new();
        let mut rng = rand::thread_rng();
        for _ in 0..output_neurons {
            neurons.push(Neuron::random(&mut rng, input_neurons));
        }

        Self { neurons }
    }
}

impl Neuron {
    fn propagate(&self, inputs: &[f32]) -> f32 {

        assert_eq!(inputs.len(), self.weights.len());

        let mut output = 0.0;

        output = inputs
            .iter()
            .zip(&self.weights)
            .map(|(input, weight)| input * weight)
            .sum::<f32>();
    
        (self.bias + output).max(0.0)
    }
    pub fn random(rng: &mut dyn rand::RngCore, output_size: usize) -> Self {

        let bias = rng.gen_range(-1.0..=1.0);

        let weights = (0..output_size)
            .map(|_| rng.gen_range(-1.0..=1.0))
            .collect();

        Self { bias, weights }
    }
}


mod tests {
    use super::*;
    mod random {
        use super::*;

        #[test]
        fn test() {
            let mut rng = ChaCha8Rng::from_seed(Default::default());
            let neuron = Neuron::random(&mut rng, 4);
        
            approx::assert_relative_eq!(neuron.bias, -0.6255188);
            approx::assert_relative_eq!(neuron.weights.as_slice(), [
                0.67383957,
                0.8181262,
                0.26284897,
                0.5238807,
            ].as_ref());
        }
        
    }

    mod propagate {

        use super::*;

        #[test]
        fn test() {
            let neuron = Neuron {
                bias: 0.5,
                weights: vec![-0.3, 0.8]
            };

            approx::assert_relative_eq!(
                neuron.propagate(&[-10.0, -10.0]),
                0.0
            );

            approx::assert_relative_eq!(
                neuron.propagate(&[0.5, 1.0]),
                (-0.3 * 0.5) + (0.8 * 1.0) + 0.5
            );
        }
    }
}