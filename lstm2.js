function sig(x) { return 1/(1+Math.exp(-x)); }
function tanh(x){ return Math.tanh(x); }

// weights for gates (2 inputs + 1 hidden = 3 inputs per gate)
let wf1=Math.random(), wf2=Math.random(), wfH=Math.random(), bf=Math.random();
let wi1=Math.random(), wi2=Math.random(), wiH=Math.random(), bi=Math.random();
let wc1=Math.random(), wc2=Math.random(), wcH=Math.random(), bc=Math.random();
let wo1=Math.random(), wo2=Math.random(), woH=Math.random(), bo=Math.random();

let h = 0;   // hidden
let c = 0;   // cell state
let lr = 0.1;

let data = [
  {x:[0.2,0.3], target:0.5},
  {x:[0.5,0.5], target:1.0},
  {x:[0.1,0.9], target:1.0}
];

for(let epoch=0; epoch<200; epoch++){
  for(let d of data){
    let x1 = d.x[0], x2 = d.x[1], target = d.target;

    // --- forward ---
    let f = sig(wf1*x1 + wf2*x2 + wfH*h + bf);       // forget gate
    let i = sig(wi1*x1 + wi2*x2 + wiH*h + bi);       // input gate
    let c_tilde = tanh(wc1*x1 + wc2*x2 + wcH*h + bc);// candidate
    c = f*c + i*c_tilde;                             // new cell
    let o = sig(wo1*x1 + wo2*x2 + woH*h + bo);       // output gate
    h = o * tanh(c);                                 // new hidden

    let out = h;
    let error = out - target;

    // --- backward (super simplified) ---
    // (not a full LSTM backprop, just dummy gradient update to keep style)
    wf1 -= lr*error*x1; wf2 -= lr*error*x2; wfH -= lr*error*h; bf -= lr*error;
    wi1 -= lr*error*x1; wi2 -= lr*error*x2; wiH -= lr*error*h; bi -= lr*error;
    wc1 -= lr*error*x1; wc2 -= lr*error*x2; wcH -= lr*error*h; bc -= lr*error;
    wo1 -= lr*error*x1; wo2 -= lr*error*x2; woH -= lr*error*h; bo -= lr*error;
  }
}

// ✅ final test
let x1=0.2, x2=0.3;
let f = sig(wf1*x1 + wf2*x2 + wfH*h + bf);
let i = sig(wi1*x1 + wi2*x2 + wiH*h + bi);
let c_tilde = tanh(wc1*x1 + wc2*x2 + wcH*h + bc);
c = f*c + i*c_tilde;
let o = sig(wo1*x1 + wo2*x2 + woH*h + bo);
h = o * tanh(c);
console.log("Final LSTM output:", h.toFixed(4));
