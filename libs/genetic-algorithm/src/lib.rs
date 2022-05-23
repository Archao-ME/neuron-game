use rand::RngCore;
use rand::seq::SliceRandom;

mod chromosome;

pub use self:: {
    chromosome::*
};

pub struct GeneticAlgorithm<S> {
    selection_method: S,
    crossover_method: Box<dyn CrossoverMethod>
}

pub trait Individual {
    fn fitness(&self) -> f32;

    fn chromosome(&self) -> &Chromosome;
}

pub trait SelectionMethod {
    fn select<'a, I>(
        &self, 
        rng: &mut dyn RngCore,
        population: &'a [I]
    ) -> &'a I
    where
        I: Individual;
}

impl<S> GeneticAlgorithm<S>
where 
    S: SelectionMethod,
{
    pub fn new(
        selection_method: S,
        crossover_method: impl CrossoverMethod + 'static
    ) -> Self {
        Self { 
            selection_method,
            crossover_method: Box::new(crossover_method)
         }
    }

    pub fn evolve<I>(
        &self,
        rng: &mut dyn RngCore,
        population: &[I]
    ) -> Vec<I>
    where
        I: Individual,
        {
            (0..population.len())
                .map(|_| {
                    let parent_a = self
                        .selection_method
                        .select(rng, population)
                        .chromosome();

                    let parent_b = self
                        .selection_method
                        .select(rng, population)
                        .chromosome();
                    
                    let mut child = self
                        .crossover_method
                        .crossover(rng, parent_a, parent_b);
                    
                        // TODO mutation
                        // TODO convert `Chromosome` back into `Individual`
                        todo!()
                })
                .collect()
        }

}

pub struct RouletteWheelSelection;

impl RouletteWheelSelection {
    pub fn new() -> Self {
        Self
    }
}

impl SelectionMethod for RouletteWheelSelection {
    fn select<'a, I>(
        &self, 
        rng: &mut dyn RngCore,
        population: &'a [I]
    ) -> &'a I 
    where 
        I: Individual, 
    {
        population.choose_weighted(rng, |individual| individual.fitness())
            .expect("got an empty population")
    }
}

#[cfg(test)]
#[derive(Clone, Debug)]
pub struct TestIndividual {
    fitness: f32
}

#[cfg(test)]
impl TestIndividual {
    pub fn new(fitness: f32) -> Self {
        Self { fitness }
    }
}
#[cfg(test)]
impl Individual for TestIndividual {
    fn fitness(&self) -> f32 {
        self.fitness
    }
    fn chromosome(&self) -> &Chromosome {
        panic!("not supported for TestIndividual")
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::collections::BTreeMap;

    use super::*;

    #[test]
    fn test() {
        let method = RouletteWheelSelection::new();
        let mut rng = ChaCha8Rng::from_seed(Default::default());

        let population = vec![
            TestIndividual::new(2.0),
            TestIndividual::new(1.0),
            TestIndividual::new(4.0),
            TestIndividual::new(3.0)
        ];

        let mut actual_histogram = BTreeMap::new();
        for _ in 0..1000 {
            let fitness = method
                .select(&mut rng, &population)
                .fitness() as i32;

            *actual_histogram
                .entry(fitness)
                .or_insert(0) += 1;
        }

        let expected_histogram = BTreeMap::from_iter(vec![
            (1, 98),
            (2, 202),
            (3, 278),
            (4, 422),
        ]);
        
        assert_eq!(actual_histogram, expected_histogram);
    }
}
