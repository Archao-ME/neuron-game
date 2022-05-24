use std::{ops::Index};

use rand::RngCore;
use rand::Rng;

#[derive(Clone, Debug)]
pub struct Chromosome {
    pub genes: Vec<f32>
}

impl Chromosome {
    pub fn len(&self) -> usize {
        self.genes.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &f32> {
        self.genes.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        self.genes.iter_mut()
    }
}

#[cfg(test)]
impl PartialEq for Chromosome {
    fn eq(&self, other: &Self) -> bool {
        approx::relative_eq!(
            self.genes.as_slice(),
            other.genes.as_slice(),
        )
    }
}

impl Index<usize> for Chromosome {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.genes[index]
    }
}

impl FromIterator<f32> for Chromosome {
    fn from_iter<T: IntoIterator<Item = f32>>(iter: T) -> Self {
        Self { genes: iter.into_iter().collect() }
    }
}

impl IntoIterator for Chromosome {
    type Item = f32;

    type IntoIter = std::vec::IntoIter<f32>;

    fn into_iter(self) -> Self::IntoIter {
        self.genes.into_iter()
    }
}

#[cfg(test)]
mod chromosome_tests {
    use super::*;

    fn chromosome() -> Chromosome {
        Chromosome { genes: vec![3.0, 1.0, 2.0] }
    }

    mod into_iter {
        use super::*;

        #[test]
        fn test() {
            let chromosome = Chromosome {
                genes: vec![3.0, 1.0, 2.0]
            };
            let genes: Vec<_> = chromosome.into_iter().collect();

            assert_eq!(genes.len(), 3);
            assert_eq!(genes[0], 3.0);
        }
    }

    mod from_iterator {
        use super::*;

        #[test]
        fn test() {
            let chromosome: Chromosome = 
                vec![3.0, 1.0, 2.0]
                    .into_iter()
                    .collect();

            assert_eq!(chromosome[0], 3.0);
            assert_eq!(chromosome[1], 1.0);
        }
    }

    mod len {
        use super::*;

        #[test]
        fn test() {
            let chromosome = chromosome();
            assert_eq!(chromosome.len(), 3);
            assert_eq!(chromosome[0], 3.0);
        }
    }

    mod iter {
        use super::*;

        #[test]
        fn test() {
            let chromosome = chromosome();

            let genes: Vec<_> = chromosome.iter().collect();

            assert_eq!(genes.len(), 3);
            assert_eq!(genes[0], &3.0);
            assert_eq!(genes[1], &1.0);
            assert_eq!(genes[2], &2.0);
        }
    }

    mod iter_mut {
        use super::*;

        #[test]
        fn test() {
            let mut chromosome = chromosome();

            chromosome.iter_mut().for_each(|gene| {
                *gene *= 10.0;
            });

            let genes: Vec<_> = chromosome.iter().collect();

            assert_eq!(genes.len(),3);
            assert_eq!(genes[0], &30.0);
            assert_eq!(genes[1], &10.0);
        }
    }
}


pub trait CrossoverMethod {
    fn crossover(
        &self,
        rng: &mut dyn RngCore,
        parent_a: &Chromosome,
        parent_b: &Chromosome
    ) -> Chromosome;
}

#[derive(Clone, Debug)]
pub struct UniformCrossover;

impl UniformCrossover {
    pub fn new() -> Self {
        Self
    }
}

impl CrossoverMethod for UniformCrossover {
    fn crossover(
        &self,
        rng: &mut dyn RngCore,
        parent_a: &Chromosome,
        parent_b: &Chromosome
    ) -> Chromosome {
        assert_eq!(parent_a.len(), parent_b.len());
        let parent_a = parent_a.iter();
        let parent_b = parent_b.iter();

        parent_a
            .zip(parent_b)
            .map(|(&a, &b)| if rng.gen_bool(0.5) { a } else { b })
            .collect()
    }
}

#[cfg(test)]
mod crossover_test {

    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test() {
        let mut rng = ChaCha8Rng::from_seed(Default::default());
        let parent_a = (1..100)
            .map(|n| n as f32)
            .collect();

        let parent_b = (1..100)
            .map(|n| -n as f32)
            .collect();

        let child = UniformCrossover::new()
            .crossover(&mut rng, &parent_a, &parent_b);

        let diff_a = child
            .iter()
            .zip(parent_a)
            .filter(|(c, p)| *c != p)
            .count();

        let diff_b = child
            .iter()
            .zip(parent_b)
            .filter(|(c, p)| *c != p)
            .count();

        assert_eq!(diff_a, 49);
        assert_eq!(diff_b, 50);
    }
}


pub trait MutationMethod {
    fn mutate(&self, rng: &mut dyn RngCore, child: &mut Chromosome);
}

#[derive(Clone, Debug)]
pub struct GaussianMutation {
    chance: f32,

    coeff: f32,
}

impl GaussianMutation {

    pub fn new(chance: f32, coeff: f32) -> Self {
        assert!(chance >= 0.0 && chance <= 1.0);

        Self { chance, coeff }
    }
    
}

impl MutationMethod for GaussianMutation {
    fn mutate(&self, rng: &mut dyn RngCore, child: &mut Chromosome) {
        for gene in child.iter_mut() {
            let sign = if rng.gen_bool(0.5) { -1.0 } else { 1.0 };

            if rng.gen_bool(self.chance as _) {
                *gene += sign * self.coeff * rng.gen::<f32>();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::ChaCha8Rng;
    use rand::SeedableRng;

    fn actual(chance: f32, coeff: f32) -> Vec<f32> {
        let mut child = vec![1.0, 2.0, 3.0, 4.0, 5.0]
            .into_iter()
            .collect();

        let mut rng = ChaCha8Rng::from_seed(Default::default());

        GaussianMutation::new(chance, coeff)
            .mutate(&mut rng, &mut child);
        
            child.into_iter().collect()
    }

    mod given_zero_chance {

        fn actual(coeff: f32) -> Vec<f32> {
            super::actual(0.0, coeff)
        }
    
        mod and_zero_coefficient {
            use super::*;
    
            #[test]
            fn does_not_change_the_original_chromosome() {
                let actual = actual(0.0);
                let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    
                approx::assert_relative_eq!(
                    actual.as_slice(),
                    expected.as_slice(),
                );
            }
        }

        mod and_nonzero_coefficient {
            use super::*;
    
            #[test]
            fn does_not_change_the_original_chromosome() {
                let actual = actual(0.5);
                let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    
                approx::assert_relative_eq!(
                    actual.as_slice(),
                    expected.as_slice(),
                );
            }
        }

    }
}