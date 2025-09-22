// 3x3 image
let image = [
  [1,0,1],
  [0,1,0],
  [1,0,1]
];

// 2x2 filter
let f11 = Math.random();
let f12 = Math.random();
let f21 = Math.random();
let f22 = Math.random();

let target = 1;
let lr = 0.1;

for (let epoch = 0; epoch < 50; epoch++) {
  // --- forward ---
  let z11 = image[0][0]*f11 + image[0][1]*f12 + image[1][0]*f21 + image[1][1]*f22;
  let z12 = image[0][1]*f11 + image[0][2]*f12 + image[1][1]*f21 + image[1][2]*f22;
  let z21 = image[1][0]*f11 + image[1][1]*f12 + image[2][0]*f21 + image[2][1]*f22;
  let z22 = image[1][1]*f11 + image[1][2]*f12 + image[2][1]*f21 + image[2][2]*f22;

  let h11 = Math.max(0,z11);
  let h12 = Math.max(0,z12);
  let h21 = Math.max(0,z21);
  let h22 = Math.max(0,z22);

  let out = h11 + h12 + h21 + h22;

  let error = out - target;

  // --- backward (simple update) ---
  f11 -= lr * error;
  f12 -= lr * error;
  f21 -= lr * error;
  f22 -= lr * error;
}

// ✅ final prediction
let z11 = image[0][0]*f11 + image[0][1]*f12 + image[1][0]*f21 + image[1][1]*f22;
let z12 = image[0][1]*f11 + image[0][2]*f12 + image[1][1]*f21 + image[1][2]*f22;
let z21 = image[1][0]*f11 + image[1][1]*f12 + image[2][0]*f21 + image[2][1]*f22;
let z22 = image[1][1]*f11 + image[1][2]*f12 + image[2][1]*f21 + image[2][2]*f22;

let h11 = Math.max(0,z11);
let h12 = Math.max(0,z12);
let h21 = Math.max(0,z21);
let h22 = Math.max(0,z22);

let finalOut = h11 + h12 + h21 + h22;

console.log("Final CNN output:", finalOut.toFixed(4));
