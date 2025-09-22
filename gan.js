function sig(x){
  return 1/(1+Math.exp(-x)); 
}

let wg = Math.random();
let wd = Math.random();
let bd = Math.random();

let target = 2;
let lr = 0.1;

for (let epoch=0; epoch<200; epoch++) {
  // --- real ---
  let real = target;
  let zR = wd*real + bd;
  let dReal = sig(zR);

  // --- fake ---
  let z = Math.random();
  let g = wg*z;
  let zF = wd*g + bd;
  let dFake = sig(zF);

  // --- discriminator error ---
  let dErr = (dReal - 1) + (dFake - 0);

  // update D
  wd -= lr*dErr;
  bd -= lr*dErr;

  // --- generator error ---
  let gErr = (dFake - 1);

  // update G
  wg -= lr*gErr*z;
}

// ✅ final outputs after training 
let testZ = Math.random();
let genImage = wg * testZ; // generator’s final fake output
let dOnFake = sig(wd*genImage + bd);
let dOnReal = sig(wd*target + bd);

console.log("Final Generator output (fake image):", genImage.toFixed(4));
console.log("Discriminator on Real:", dOnReal.toFixed(4));
console.log("Discriminator on Fake:", dOnFake.toFixed(4));
