use rand::RngCore;
use rand::seq::SliceRandom;

mod chromosome;

pub use self:: {
    chromosome::*
};

pub struct GeneticAlgorithm<S> {
    selection_method: S,
    crossover_method: Box<dyn CrossoverMethod>,
    mutation_method: Box<dyn MutationMethod>
}

pub trait Individual {
    fn create(chromosome: Chromosome) -> Self;
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
        crossover_method: impl CrossoverMethod + 'static,
        mutation_method: impl MutationMethod + 'static
    ) -> Self {
        Self { 
            selection_method,
            crossover_method: Box::new(crossover_method),
            mutation_method: Box::new(mutation_method)
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
                    
                    self.mutation_method.mutate(rng, &mut child);

                    I::create(child)
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
#[derive(Clone, Debug, PartialEq)]
pub enum TestIndividual {
    WithChromosome { chromosome: Chromosome },
    WithFitness { fitness: f32 },
}

#[cfg(test)]
impl TestIndividual {
    pub fn new(fitness: f32) -> Self {
        Self::WithFitness { fitness }
    }
}
#[cfg(test)]
impl Individual for TestIndividual {
    fn create(chromosome: Chromosome) -> Self {
        Self::WithChromosome { chromosome }
    }
    fn chromosome(&self) -> &Chromosome {
        match self {
            Self::WithChromosome { chromosome } => chromosome,
            Self::WithFitness { .. } => {
                panic!("...")
            }
        }
    }
    fn fitness(&self) -> f32 {
        match self {
            Self::WithChromosome { chromosome } => {
                chromosome.iter().sum()
            }

            Self::WithFitness { fitness } => *fitness
        }
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


#[cfg(test)]
mod population_expected {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn individual(genes: &[f32]) -> TestIndividual {
        let chromosome = genes.iter().cloned().collect();
    
        TestIndividual::create(chromosome)
    }

    #[test]
    fn test() {
        let mut rng = ChaCha8Rng::from_seed(Default::default());

        let ga = GeneticAlgorithm::new(
            RouletteWheelSelection::new(),
            UniformCrossover::new(),
            GaussianMutation::new(0.5, 0.5),
        );

        let mut population = vec![
            /* todo */
            individual(&[0.0, 0.0, 0.0]), // fitness = 0.0
            individual(&[1.0, 1.0, 1.0]), // fitness = 3.0
            individual(&[1.0, 2.0, 1.0]), // fitness = 4.0
            individual(&[1.0, 2.0, 4.0]), // fitness = 7.0
        ];

        for _ in 0..10 {
            population = ga.evolve(&mut rng, &population);
        }

        let expected_population = vec![
            individual(&[0.44769490, 2.0648358, 4.3058133]),
            individual(&[1.21268670, 1.5538777, 2.8869110]),
            individual(&[1.06176780, 2.2657390, 4.4287640]),
            individual(&[0.95909685, 2.4618788, 4.0247330]),
        ];

        assert_eq!(population, expected_population);
    }
}