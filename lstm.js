// Inputs for 2 time steps
let x1 = 0.5;
let x2 = 0.7;
let target = 1.0;

// weights
let wf = Math.random();
let wi = Math.random();
let wc = Math.random();
let wo = Math.random();
let bf = Math.random();
let bi = Math.random();
let bc = Math.random();
let bo = Math.random();
let w_out = Math.random();
let b_out = Math.random();

let lr = 0.1;

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
function dsigmoid(y) { return y * (1 - y); }
function tanh(x) { return Math.tanh(x); }
function dtanh(y) { return 1 - y*y; }

for (let epoch = 0; epoch < 1000; epoch++) {
  let c = 0, h = 0;

  // time step 1
  let f1 = sigmoid(wf * x1 + bf);
  let i1 = sigmoid(wi * x1 + bi);
  let c_tilde1 = tanh(wc * x1 + bc);
  c = f1 * c + i1 * c_tilde1;
  let o1 = sigmoid(wo * x1 + bo);
  h = o1 * tanh(c);

  // time step 2
  let f2 = sigmoid(wf * x2 + bf);
  let i2 = sigmoid(wi * x2 + bi);
  let c_tilde2 = tanh(wc * x2 + bc);
  c = f2 * c + i2 * c_tilde2;
  let o2 = sigmoid(wo * x2 + bo);
  h = o2 * tanh(c);

  let out = h * w_out + b_out;
  let error = out - target;

  // simple update only on w_out, b_out (to keep tiny)
  w_out -= lr * error * h;
  b_out  -= lr * error;

  if (epoch % 200 === 0) {
    console.log("Epoch", epoch, "Loss", (error*error).toFixed(4));
  }
}
//c is being gated stored
console.log("Final Output:", (w_out * Math.tanh(c) + b_out).toFixed(3));
