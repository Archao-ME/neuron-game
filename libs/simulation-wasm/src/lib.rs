use serde::Serialize;
use wasm_bindgen::prelude::*;
use lib_simulation as sim;
use rand::prelude::*;

#[wasm_bindgen]
pub struct Simulation {
    rng: ThreadRng,
    sim: sim::Simulation
}

#[wasm_bindgen]
impl Simulation {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        println!(".........new simulation");
        let mut rng = thread_rng();
        let sim = sim::Simulation::random(&mut rng);
    
        Self { rng, sim }
    }
    pub fn world(&self) -> JsValue {
        let world = World::from(self.sim.world());
        JsValue::from_serde(&world).unwrap()
    }

    pub fn step(&mut self) {
        self.sim.step();
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct World {
    pub animals: Vec<Animal>
}

#[derive(Clone, Debug, Serialize)]
pub struct Animal {
    pub x: f32,
    pub y: f32,
    pub rotation: f32
}

impl From<&sim::World> for World {
    fn from(world: &sim::World) -> Self {
        let animals = world
            .animals()
            .iter()
            .map(Animal::from)
            .collect();
        println!("........World form");
        
        Self { animals }
    }
}

impl From<&sim::Animal> for Animal {
    fn from(animal: &sim::Animal) -> Self {
        println!("...............");
        println!("debug----------{}", animal.rotation().angle());
        Self {
            x: animal.position().x,
            y: animal.position().y,
            rotation: animal.rotation().angle()
        }
    }
}