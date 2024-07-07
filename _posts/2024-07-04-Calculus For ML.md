---
title: Calculus for ML
date: 2024-07-05 10:31:00 +/-0000
categories: [ml, deep_learning]
tags: [calculus, ml]     # TAG names should always be lowercase
math: true
---
## Calculus for ML
This is mostly just a place for me to write down my thoughts so that I don't forget everything. It likely won't be very coherent, but it helps me.

## Derivatives
*Definition*: 

$$
    \lim_{h\to 0} \frac{f(x+h)-f(x)}{h} 
$$

(I promise I didn't just spend an hour figuring out how LaTeX works)

But what does this actually mean?
# Example
Given the equation ```f(x) = 2x^2```, we can take its derivative by plugging the function into the equation.

Plugging in ```f(x)```:

```lim h-> 0 (2(x+h)^2 - 2(x)^2) / h```

Expanding ```(x+h) ^2``` to ```(x^2 + 2xh + h^2)```:

```lim h-> 0 (2(x^2 + 2xh + h^2) - 2x^2) / h```

Distributing the 2 and rearranging:

```lim h-> 0 (2x^2 + 2h^2 + 4xh - 2x^2) / h```

Removing the ```2x^2``` and ```- 2x^2``` because they cancel out:

```lim h-> 0 (2h^2 + 4xh) / h```

Factoring by Greatest Common Factor (```2h```):

```lim h-> 0 (2h(h + 2x)) / h```

Simplifying the fraction (```h```s on top and bottom cancel out):

```lim h-> 0 2(h + 2x)```

Distributing the ```2```:

```lim h-> 0 2h + 4x```

As h approaches 0, the 2h term will approach 0 and we will be left with just ```4x```.

# Challenge:
Take the derivative of the function ```f(x) = x^3```

# Notation
The derivative of a function is often shown with ```df/dx``` where f is the function and x is the parameter we are using to find the derivative.

For example, with the function ```f(x) = 4x```, ```df/dx = 4```.

Derivatives are also sometimes shown using ```f'(x)``` (pronounced "f prime of x").
I will use the first notation (```df/dx```) to represent taking the derivative because it is a more accurate representation.

## Multi-Variable
Sometimes, functions can take more than one parameter. For example, in the function ```f(a, b, c) = 2a - b + c^2```, f takes three parameters: ```a```, ```b```, and ```c```. This function is evaluated like any other, just replacing the variables with their parameters.

```f(1, 2, 3) = 2(1) -(2) + (3)^2 = 9```

To take the derivate of these functions, we must choose one parameter we want to focus on. We take the derivative of the function with respect to the parameter. This is done by treating all the other variables as constants when taking the derivative, and only adding ```h``` to the variable you want to take the derivative with.

Let's try this on our function ```f``` by taking its derivative with respect to ```a```.

Plugging into the derivative equation:

```lim h -> 0 (f(a + h, b, c) - f(a, b, c)) / h```

Using the actual function:

```lim h -> 0 ((2(a+h) - b + c^2) - (2a - b + c^2)) / h```

Distributing:

```lim h -> 0 (2a + 2h -b + c^2 -2a + b -c^2) / h```

Canceling out terms:

```lim h -> 0 2h/h```

Simplifying:

```lim h -> 0 2 => 2```

```∂f/∂a = 2```

Note that I used ```∂f/∂a``` instead of ```df/dx```. There are two important changes in notation for derivatives of functions with multiple variables (partial derivatives). The first is that the symbol ```∂``` is used instead of ```d```, which just denotes that the partial derivative was taken. The second change is simply replacing ```dx``` with whatever variable you choose to take the derivative with (in our case ```∂f/∂a```). 

Now we can take the derivative of functions that take multiple parameters. Why is this useful? That's a great question.

## Uses and Visualization
<img src="/assets/DerivativeSlope.png" alt="Visualization of the derivative and a function's slope" width="500"/>

Above is a masterpiece I whipped up in MS Paint to show the relationship between a function's derivative and its slope at any given point. To find the slope of the function at a point, simply evaluate its derivative at the point. In the image, the red line represents the derivative of ```f(x)```. Where the red line is less than zero, the slope of ```f(x)``` is negative. Where it is greater than zero, the slope of ```f(x)``` is positive. When the derivative is equal to zero, the function has a minimum, as it is changing from decreasing to increasing. The green points and line represent the derivative of the function at a point and the corresponding slope of the real function.

But why are derivatives useful for Machine Learning?

The answer is that they can be used to find minimum values in functions. What does that mean? Here is an example:

If we take our function ```f(a, b, c)``` from before, we can change ```a```, ```b```, and ```c``` in a way that will get f to output a very low value by using a technique called "Gradient Descent."

We can define the "Gradient" of any parameter of a function as the partial derivative with respect to the parameter evaluated at the current point. In our previous example, ```∂f/∂a``` was equal to ```2```, which means that for any input values of ```a```, ```b```, and ```c```, ```a```'s gradient with respect to ```f``` will be ```2```. Similarly, ```∂f/∂c``` is equal to ```2c``` (Challenge: verify this on your own). This means that the gradient of ```c``` with respect to ```f``` will always be equal to ```2 * c```. 

To perform Gradient Descent on ```f```, we start by initializing ```a```, ```b```, and ```c``` to reasonable values of our choice. We then take the partial derivative of ```f``` with respect to each parameter. To find the gradients of ```a```, ```b```, and ```c```, we just plug their current values into the partial derivative functions and note the outputs. Then, for each parameter, subtract its gradient (multiplied by a small number) from its value, and repeat the process over and over until ```f``` outputs a sufficiently low value with the inputs (```a```, ```b```, ```c```).

Below is a snippet of python code to do this (using a different function ```f```):
```py
# initialize a, b, and c to random values
a = 1
b = 2

# define our function we want to minimize (quadratic so it can't just go negative)
def f(a, b):
  return (a - b) ** 2

# small number we will multiply the gradients by
change_amount = 0.01

# perform gradient descent
for step in range(100):
  # partial derivatives for each parameter
  dfda = 2 * (a - b) * 1
  dfdb = 2 * (a - b) * -1

  # update parameters
  a -= dfda * change_amount
  b -= dfdb * change_amount

print(f(a, b)) # basically zero
```

This is nice, but it's not very useful on its own. The real power comes from using it to minimize a loss function. A loss function in Machine Learning is a function that takes in a given set of inputs, their real labels, and predicted outputs from a model, as well as the model's parameters, and outputs a number representing how bad the model was at predicting the outputs. 

One famous loss function is Mean Squared Error (MSE). The function definition is very confusing, but I'll put it here anyway:

$$
\mathit{MSE}=\frac{1}{n} \sum_{i=1}^n (x_{i}-y_{i})^2 \\[2ex]
\begin{array}{@{} l >{$}l<{$} @{}} \end{array}
$$
Where $n$ is the number of examples, $x$ is the model's predicted output, and $y$ is the expected output.

This is really just taking the average distance between what the model predicted and what it was supposed to predict and squaring the result.

Note that it also takes in the given inputs and the model's parameters as inputs for optimization. We use Gradient Descent to minimize the MSE loss function by shifting the parameters of our model in the right way.

This is likely very confusing, so let's illustrate it with a quick python example. I will create a linear function to be our model that takes in the parameters ```x```, ```m```, and ```b``` (as in ```y=mx+b```). I will then create a loss function that takes in inputs, expected outputs, and our model's parameters, and spits out a number to show how wrong the model is. I will optimize ```m``` and ```b``` using gradient descent, so that the loss is as low as possible and the model fits the data well. This process is called linear regression.


