import * as sim from "lib-simulation-wasm";

console.log("debug sim", sim)

CanvasRenderingContext2D.prototype.drawTriangle = function (x, y, size, rotation) {
    this.beginPath();

    this.moveTo(
        x + Math.cos(rotation) * size * 1.5,
        y + Math.sin(rotation) * size * 1.5,
    );

    this.lineTo(
        x + Math.cos(rotation + 2.0 / 3.0 * Math.PI) * size,
        y + Math.sin(rotation + 2.0 / 3.0 * Math.PI) * size,
    );

    this.lineTo(
        x + Math.cos(rotation + 4.0 / 3.0 * Math.PI) * size,
        y + Math.sin(rotation + 4.0 / 3.0 * Math.PI) * size,
    );

    this.lineTo(
        x + Math.cos(rotation) * size * 1.5,
        y + Math.sin(rotation) * size * 1.5,
    );

    this.stroke();

};

const simulation = new sim.Simulation();
console.log("debug simulation", simulation);
const world = simulation.world();

const viewport = document.getElementById('viewport');
const viewportScale = window.devicePixelRatio || 1;
const viewportWidth = viewport.width;
const viewportHeight = viewport.height;
viewport.width = viewportWidth * viewportScale;
viewport.height = viewportHeight * viewportScale;
viewport.style.width = viewportWidth + 'px';
viewport.style.height = viewportHeight + 'px';

const ctxt = viewport.getContext('2d');
ctxt.scale(viewportScale, viewportScale);

ctxt.fillStyle = 'rgb(0, 0, 0)';

for (const animal of world.animals) {
    ctxt.drawTriangle(
        animal.x * viewportWidth,
        animal.y * viewportHeight,
        0.01 * viewportWidth,
        animal.rotation
    );
}

function redraw() {
    ctxt.clearRect(0, 0, viewportWidth, viewportHeight);

    simulation.step();

    for (const animal of simulation.world().animals) {
        ctxt.drawTriangle(
            animal.x * viewportWidth,
            animal.y * viewportHeight,
            0.01 * viewportWidth,
            animal.rotation,
        );
    }

    requestAnimationFrame(redraw);
}

redraw();