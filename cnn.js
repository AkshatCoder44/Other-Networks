// --- 2x2 image ---
let image = [
  [1, 0],
  [0, 1]
];

// --- 2x2 filter (weights) ---
let f11 = Math.random();
let f12 = Math.random();
let f21 = Math.random();
let f22 = Math.random();

// target output
let target = 1;
let lr = 0.1;

// --- training ---
for (let epoch = 0; epoch < 50; epoch++) {
  // forward: single convolution (since filter covers whole 2x2 image)
  let z = image[0][0]*f11 + image[0][1]*f12 +
          image[1][0]*f21 + image[1][1]*f22;

  let h = Math.max(0, z);   // ReLU activation
  let out = h;              // output layer (no extra weights here)

  // error
  let error = out - target;

  // backward: update filter weights
  f11 -= lr * error * image[0][0];
  f12 -= lr * error * image[0][1];
  f21 -= lr * error * image[1][0];
  f22 -= lr * error * image[1][1];
}

// --- final prediction ---
let z = image[0][0]*f11 + image[0][1]*f12 +
        image[1][0]*f21 + image[1][1]*f22;

let h = Math.max(0, z);
let finalOut = h;

console.log("Final CNN output:", finalOut.toFixed(4));
