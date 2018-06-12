let points_x = [], points_y = [], xs = [], ys = [];
const xmin = -2, ymin = -2, xmax = 2, ymax = 2; //Canvas boundaries
const point_radius = 10;
const resolution = 0.025;

const stddv = 0.5;
const a = tf.variable(tf.randomNormal([1],0,stddv));
const b = tf.variable(tf.randomNormal([1],0,stddv));
const c = tf.variable(tf.randomNormal([1],0,stddv));
const d = tf.variable(tf.randomNormal([1],0,stddv));

const learningRate = 0.1;
const optimizer = tf.train.adam(learningRate);

function setup()
{
  createCanvas(600, 400);
  background(50);
  noStroke();

  for (let i = xmin; i <= xmax + resolution; i = i + resolution) //Creating xs for plotting the guessed polynomial
  {
    xs.push(i);
  }
  xs = tf.tensor1d(xs);
}

function draw() 
{
  background(50);

  noFill();
  stroke(150);
  strokeWeight(4);
  ys = predict(xs); //Estimating ys for plotting the guessed polynomial 
  beginShape(); //Plotting the guess
  for (let i = 0; i < xs.size; i++)
  {
    vertex(map(xs.dataSync()[i],xmin,xmax,0,width),map(ys.dataSync()[i],ymin,ymax,height,0));
  }
  endShape();
  ys.dispose(); //Dispose tensor from memory

  fill(200);
  noStroke();
  for(let i =0; i < points_x.length; i++) //Plotting the points
  {
    ellipse(map(points_x[i],xmin,xmax,0,width),map(points_y[i],ymin,ymax,height,0),point_radius,point_radius);
  }

  textSize(20);
  fill(225); // vv  Writing the guessed coefficients  vv
  text(a.dataSync()[0].toFixed(2) + ' x^3 + ' + b.dataSync()[0].toFixed(2) + ' x^2 + ' + c.dataSync()[0].toFixed(2) + ' x + ' + d.dataSync()[0].toFixed(2),30,30);

  if (points_x.length != 0) //Make a training step if there's at least one point
  {
    optimizer.minimize(() => {
      const predictions = predict(tf.tensor1d(points_x));
      return mse(predictions,tf.tensor1d(points_y));
    },false,[a,b,c,d]);
  }
}

function mouseDragged()//Add points when you drag the mouse mousePressed could also be used
{
  points_x.push(map(mouseX,0,width,xmin,xmax));
  points_y.push(map(mouseY,0,height,ymax,ymin));
}

function predict(x)
{
  return tf.tidy(() => {
    return a.mul(x.pow(tf.scalar(3)))//.mul(alpha_a)
      .add(b.mul(x.square()))//.mul(alpha_b)
      .add(c.mul(x))//.mul(alpha_c)
      .add(d)
  });
}

function mse(predictions, labels) //Mean Squared Error
{
  return tf.tidy(() => {
    return predictions.sub(labels).square().mean();
  })
}