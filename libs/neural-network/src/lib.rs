#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}

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

impl Network {
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
}

impl Neuron {
    fn propagate(&self, inputs: &Vec<f32>) -> f32 {

        assert_eq!(inputs.len(), self.weights.len());

        let mut output = 0.0;

        output = inputs
            .iter()
            .zip(&self.weights)
            .map(|(input, weight)| input * weight)
            .sum::<f32>();
    
        (self.bias + output).max(0.0)
    }
}