# Pytorch Notes


## Tensor

The tensor is the central data structure used in pytorch. It represents an
n-dimensional list of data structures. A tensor is actually just a
generalization of a vector and a matrix.

### Tensor dimensions
- 0-dimensional tensor: scalar
- 1-dimensional tensor: vector 
- 2-dimensional tensor: matrix 
- 3-dimensional tensor: 3-D tensor
...
- n-dimensional tensor: n-D tensor 

### How a tensor is represented

- The following is pseudo-code that lays out the metadata a tensor contains.

```
class Tensor():
	size:   (D, W, H)    // Lays out how long each dimension is
	stride: (H*W, W, 1)  // Used for traversing tensor
	dtype:  float				 // Contains data-type of scalers
	device: cuda  			 // Contains type of device tensor is loaded on
	layout: strided      // Tells how to traverse tensor
	offset: 0						 // This is used when a Tensor is a view. it is usually 0 
```


### What is Stride?

- The problem with representing tensors, is that they can have several 
dimensions, while memory is a contiguous block.

- This problem is solved using strided representation. 
