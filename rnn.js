// Inputs for 3 time steps
let x1 = 0.2, x2 = 0.4, x3 = 0.6;
let target = 0.8; 

// weights
let w_in = Math.random();   // input → hidden
let w_hh = Math.random();   // hidden → hidden
let w_out = Math.random();  // hidden → output
let b_h = Math.random();
let b_o = Math.random();

let lr = 0.1;

// Train for few epochs
for (let epoch = 0; epoch < 1000; epoch++) {
  let h_prev = 0;
  let loss = 0;

  // time steps
  let hs = [];
  for (let t of [x1, x2, x3]) {
    let z = w_in * t + w_hh * h_prev + b_h;
    let h = Math.tanh(z);
    hs.push(h);
    h_prev = h;
  }

  let out = hs[hs.length - 1] * w_out + b_o;
  let error = out - target;
  loss += error * error;

  // backprop (very simplified)
  let dout = error;
  w_out -= lr * dout * hs[hs.length - 1];
  b_o   -= lr * dout;

  let dh = dout * w_out * (1 - hs[2] ** 2); 
  w_in -= lr * dh * x3;
  w_hh -= lr * dh * hs[1];
  b_h  -= lr * dh;

  if (epoch % 200 === 0) {
    console.log("Epoch", epoch, "Loss", loss.toFixed(4));
  }
}

console.log("Final Output:", (hs[2] * w_out + b_o).toFixed(3));
