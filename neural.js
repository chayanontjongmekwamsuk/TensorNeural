const tf = require('@tensorflow/tfjs');
const iris = require('./iris.json');

//seperating input and output and change type of data to array
const fullin = iris.map(h => [h.sepal_length,h.sepal_width,h.petal_length,h.petal_width]) 
const fullout = iris.map(h => [
    h.species === "setosa" ? 1:0,
    h.species === "virginica" ? 1:0,
    h.species === "versicolor" ? 1:0
])
 
//spliting data and packed data => 
const numtest  = Math.round(iris.length * 0.2);                     //  using 20% of data for testing
const numtrain  = iris.length - numtest;                            //  80% of data for training
const trainin = tf.tensor2d(fullin.slice(0,numtrain))
const trainout = tf.tensor2d(fullout.slice(0,numtrain))
const testin = tf.tensor2d(fullin.slice(numtrain,iris.length))
const testout = tf.tensor2d(fullout.slice(numtrain,iris.length))


const input = tf.tensor2d([4.8,3,1.5,0.2],[1,4])
// build layer 
model = tf.sequential()
    // input layer
    model.add(tf.layers.dense({
        inputShape : [4],
        activation : "sigmoid",                    // 
        units : 5                                 //unit must be 2/3 of input plus output
    }))

    // hidden layer 
    model.add(tf.layers.dense({
        inputShape : [5],
        activation : "sigmoid",         
        units : 3                                 //output have 3 units
    }))

    // outputlayer
    model.add(tf.layers.dense({
        activation : "softmax",                   // normalize output
        units : 3                           
    }))

   
 // error function 
 model.compile({
   // loss : "meanSquaredError" ,
     loss : "categoricalCrossentropy",               // great for clssification
     optimizer : tf.train.adam(.06),
     metrics : ['accuracy']
 })

 //time
 const starttime = Date.now();   
 // train 
 model.fit(trainin,trainout,{epochs :100,validation : [testin,testout]})
.then((history) =>{
    //console.log("done",Date.now()-starttime)
    console.log(history)
  model.predict(input).print()
  
  
})


/*
const newTensor = tf.tensor2d([[2,4], [5,6]])
newTensor.get([0])
newTensor.get([3]) ##returns 6
*/







   
   
   