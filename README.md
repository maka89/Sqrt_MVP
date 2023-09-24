# Sqrt_MVP
Calculates the product between square roots of matrices and vectors.
Supports using Pade Approximants + Taylor Polynomials to approximate the matrix square root.
Since these methods can get a good performance boost from only calculating the matrix-vector product and not the matrix itself.
Pade is most accurate. Taylor does not require forming any matrices during the calculation.

## Usage
```python
  N=50
  X=torch.randn(3,40,100,100,N) 
  C=torch.matmul(torch.transpose(X,-1,-2),X) # Batch of pos-def matrices
  v=torch.randn(3,40,100,N,1) # Batch of vectors
  
  # 6 terms Pade approximation. Matrix Square Root
  p = Pade(6,False)
  p.set_mat(C)
  pv = p.matvec(v) #Matrix-vector product

  # 20 terms Taylor approximation. Inverse Matrix Square Root
  t = Taylor(20,True)
  t.set_mat(C)
  tv = p.matvec(v) #Matrix-vector product
  

  

```
